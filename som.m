classdef som < RegFit.fitModel
    % Suri and Onori Model
    
    properties( SetAccess = immutable )
        Ic          double  {mustBeGreaterThanOrEqual( Ic, 0 ),...          % C-rate
                             mustBeLessThanOrEqual( Ic, 10 ),...
                             mustBeReal( Ic )}
        T           double  {mustBeGreaterThanOrEqual( T, -40 ),...         % Temperature [deg C]
                             mustBeLessThanOrEqual( T, 150 ), ...
                             mustBeReal( T )}
        SOC         double  {mustBeGreaterThanOrEqual( SOC, 0 ),...         % Average State of Charge [0, 1]
                             mustBeLessThanOrEqual( SOC, 1 ), ...
                             mustBeReal( SOC )}
    end
    
    properties
        Theta       double                                                  % Model parameter vector
        LB          double = zeros( 4, 1 );                                 % Parameter lower bounds
        UB          double = [10 10 10 1].';                                % Parameter upper bounds
    end
    
    properties ( SetAccess = protected )
        ParNames    string = ["\alpha", "\beta", "\eta", "z"]               % Parameter names
    end
    
    properties ( Constant = true )
        NumFitCoeff int8   = 4                                              % Number of parameters to estimate from the data
        Rg          double = 8.3140                                         % Universal gas constant
        Ea          double = 31500                                          % Activation energy
        ModelName   RegFit.fitModelType = "som"                             % Model name
    end
    
    properties ( SetAccess = protected, Dependent = true )
        Alpha                                                               % SOC gain multiplier
        Beta                                                                % SOC gain offset
        Eta                                                                 % Arrhenius Coefficient 
        Z                                                                   % Power Law Index
    end
    
    methods
        function obj = som( ReEstObj, SOC, Ic, T )
            %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.som( ReEstObj, SOC, Ic, T );
            %
            % Input Arguments:
            %
            % ReEstObj  --> RegFit.reEstLamdaContext object. Implements
            %               regularisation parameter re-estimation 
            %               algorithm.
            % SOC       --> Mean state of charge [0, 1].
            % Ic        --> Average C-rate
            %--------------------------------------------------------------
            obj.ReEstObj = ReEstObj;
            %--------------------------------------------------------------
            % Assign auxilary data
            %--------------------------------------------------------------
            obj.SOC = SOC;
            obj.Ic = Ic;
            obj.T = T;
        end
        
        function J = jacobean( obj, X, Beta )   
            %----------------------------------------------------------------
            % Return Jacobean matrix
            %
            % J = obj.jacobean( X, Beta );
            %
            % Input Arguments:
            %
            % X       --> Dependent data
            % Beta    --> Coefficient Vector {obj.Theta}. Assumed format:
            %             [Alpha, Beta, Eta, Z].'
            %
            % Output Arguments:
            % J       --> Jacobean: [df_dAlpha, df_dBeta, df_dEta, df_dZ]
            %----------------------------------------------------------------
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            [A, B, E, Idx] = obj.assignPars( Beta );                        % Assign the parameters
            N = numel( X );
            J = zeros( N, obj.NumFitCoeff );
            %--------------------------------------------------------------
            % Derivative wrt Alpha
            %--------------------------------------------------------------
            J( :,1 ) = obj.SOC*exp(-(obj.Ea - obj.Ic*E)/(obj.Rg*(obj.T +...
                273.15)))*X.^Idx;
            %--------------------------------------------------------------
            % Derivative wrt Beta
            %--------------------------------------------------------------
            J( :,2 ) = exp(-(obj.Ea - obj.Ic*E)/(obj.Rg*(obj.T +...
                273.15)))*X.^Idx;
            %--------------------------------------------------------------
            % Derivative wrt Eta
            %--------------------------------------------------------------
            J( :,3 ) = 1000*(A*obj.SOC + B)*obj.Ic*...
                exp(-(obj.Ea - obj.Ic*E)/(obj.Rg*(obj.T + 273.15)))/...
                (obj.Rg*(obj.T + 273.15))*X.^Idx;
            %--------------------------------------------------------------
            % Derivative wrt Z
            %--------------------------------------------------------------
            J( :,4 ) = (B + obj.SOC*A)*...
                exp(-(obj.Ea - obj.Ic*E)/(obj.Rg*(obj.T + 273.15)))*...
                log(X).*X.^Idx;
        end
        
        function  Yhat = predictions( obj, X, Beta )                                
            %--------------------------------------------------------------
            % Suri and Onori Model predictions
            %
            % Yhat = obj.predictions( X, Beta );
            %
            % Input Arguments:
            %
            % X       --> Independent data
            % Beta    --> Coefficient Vector {obj.Theta}. Assumed format:
            %             [Alpha, Beta, Eta, Z].'
            %--------------------------------------------------------------
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            [~, ~, ~, Idx] = obj.assignPars( Beta );                        % Assign the parameters
            Sfac = obj.severityFactor( Beta );                              % Calculate severity factor
            Yhat = Sfac*X.^Idx;                                             % Predictions
        end
        
        function Sfac = severityFactor( obj, Beta )
            %--------------------------------------------------------------
            % Calculate severity factor
            %
            % Sfac = obj.severityFactor( Beta );
            %
            % Input Arguments:
            %
            % Beta    --> Coefficient Vector {obj.Theta}. Assumed format:
            %             [Alpha, Beta, Eta, Z].'
            %--------------------------------------------------------------
            if ( nargin < 2 )
                Beta = obj.Theta;                                           % Apply default
            end
            [A, B, E] = obj.assignPars( Beta );                             % Assign the parameters
            Sfac = (A*obj.SOC + B)*...
                exp(( -obj.Ea + E*obj.Ic )/ obj.Rg /( obj.T + 273.15));
        end
        
        function V = startingValues( obj, X, Y )
            %--------------------------------------------------------------
            % Estimate starting parameter coefficient values
            %
            % obj = obj.startingValues( X, Y );
            %
            % Input values:
            %
            % X     --> Input data
            % Y     --> Observed response data
            %--------------------------------------------------------------
            V = zeros( obj.NumFitCoeff, 1 );
            %--------------------------------------------------------------
            % Starting value for Power Law Index
            %--------------------------------------------------------------
            P = ( X <= 0 ) | ( Y <= 0);
            Xz = X( ~P );
            Xz = [ones(size( Xz )) log10( Xz )];
            B = Xz\log10( Y( ~P ) );
            SevFac = 10^B( 1 );
            V(4) = B( 2 );                                                  % Initial value for power law index
            %--------------------------------------------------------------
            % Make the assumption that (Alpha*SOC + Beta) = 1. Then solve
            % for Eta
            %--------------------------------------------------------------
            V(3) = ( log( SevFac )*obj.Rg*( obj.T + 273.15 ) +...
                obj.Ea )/obj.Ic/1000;                                       % Initial estimate for
            %--------------------------------------------------------------
            % Initially set Beta = 0.5 and then Alpha = (1 - Beta)/SOC;
            %--------------------------------------------------------------
            V(2) = 0.5;
            V(1) = ( 1 - V(2) )./obj.SOC;
        end
        
        function obj = setCoefficientBnds( obj, LB, UB )
            %--------------------------------------------------------------
            % Set bound constraints for model fit parameters
            %
            % obj = obj.setCoefficientBnds( LB, UB );
            %
            % Input Arguments:
            %
            % LB    --> Lower bound for parameter estimates
            % UB    --> Upper bound for parameter estimates
            %--------------------------------------------------------------
            if ( numel( LB ) == obj.NumFitCoeff ) && ( numel( UB ) == obj.NumFitCoeff )
                obj.LB = LB( : );
                obj.UB = UB( : );
            else
                error('Arguments must have %2.0d elements', obj.NumFitCoeff );
            end
        end
    end % constructor and ordinary methods
    
    methods
        function A = get.Alpha( obj )
            % return SOC gain multiplier coefficient
            A = obj.Theta(1);
        end
        
        function B = get.Beta( obj )
            % return SOC gain offset coefficient
            B = obj.Theta(2);
        end
        
        function E = get.Eta( obj )
            % return Arrhenius like coefficient
            E = obj.Theta(3);
        end        
        
        function Z = get.Z( obj )
            % return power law index
            Z = obj.Theta(4);
        end        
    end % get/set methods
    
    methods ( Access = protected )     
    end % protected methods
    
    methods ( Static = true, Hidden = true )
        function [X, Y, W] = parseInputs( X, Y, W )
            %--------------------------------------------------------------
            % Remove zero or negative data
            %
            % [X, Y, W] = RegFit.som.parseInputs( X, Y, W );
            % [X, Y, W] = obj.parseInputs( X, Y, W );
            %
            % X             --> Independent data
            % Y             --> Observed data vector
            % W             --> Weights
            %--------------------------------------------------------------
            P = ( X <= 0 ) | ( Y <= 0 );
            X = X( ~P );
            Y = Y( ~P );
            W = W( ~P );
        end
        
        function [A, B, E, Z] = assignPars( Beta )
            %--------------------------------------------------------------
            % Assign the parameter vector contents
            %
            % [Alpha, Beta, Eta, Z] = obj.assignPars( Beta );
            % [Alpha, Beta, Eta, Z] = RegFit.som.assignPars( Beta );
            %--------------------------------------------------------------
            A = Beta(1);
            B = Beta(2);
            E = 1000*Beta(3);                                               % Invert scaling
            Z = Beta(4);
        end
     end % static methods
end