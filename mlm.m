classdef mlm < RegFit.fitModel
    % Martinez-Laserna Model
    
    properties( SetAccess = immutable )
        T           double  {mustBeGreaterThanOrEqual( T, -40 ),...         % Temperature [deg C]
                             mustBeLessThanOrEqual( T, 150 ), ...
                             mustBeReal( T )}
        SOC         double  {mustBeGreaterThanOrEqual( SOC, 0 ),...         % Average State of Charge [0, 1]
                             mustBeLessThanOrEqual( SOC, 1 ), ...
                             mustBeReal( SOC )}
    end
    
    properties
        Theta       double                                                  % Model parameter vector
        LB          double = [0 -1 -1 0].';                                 % Parameter lower bounds
        UB          double = [1 1 1 2].';                                   % Parameter upper bounds
    end
    
    properties ( SetAccess = protected )
        ParNames    string = ["\omega", "\beta_1", "\beta_2", "z"]          % Parameter names
    end
    
    properties ( Constant = true )
        NumFitCoeff int8   = 4                                              % Number of parameters to estimate from the data
        ModelName   RegFit.fitModelType = "mlm"                             % Model name
    end    
    
    properties ( SetAccess = protected, Dependent = true )
        Omega                                                               % Gain multiplier
        Beta_1                                                              % Arrhenius Temperature Coefficient
        Beta_2                                                              % Arrhenius SOC Coefficient 
        Z                                                                   % Power Law Index
    end    
    
    methods
        function obj = mlm( ReEstObj, SOC, T )
            %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.mlm( ReEstObj, SOC, T );
            %
            % Input Arguments:
            %
            % ReEstObj  --> RegFit.reEstLamda object. Implements
            %               regularisation parameter re-estimation 
            %               algorithm.
            % SOC       --> Mean state of charge [0, 1].
            %--------------------------------------------------------------
            obj.ReEstObj = ReEstObj;
            %--------------------------------------------------------------
            % Assign auxilary data
            %--------------------------------------------------------------
            obj.SOC = SOC;
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
            [W, B1, B2, Idx] = obj.assignPars( Beta );                        % Assign the parameters
            N = numel( X );
            J = zeros( N, obj.NumFitCoeff );
            %--------------------------------------------------------------
            % Derivative wrt Omega
            %--------------------------------------------------------------
            J( :,1 ) = exp(B1/(obj.T + 273.15) + B2*obj.SOC).*X.^Idx;
            %--------------------------------------------------------------
            % Derivative wrt Beta1
            %--------------------------------------------------------------
            J( :,2 ) = W*exp(B1/(obj.T + 273.15) + B2*obj.SOC).*X.^Idx;
            J( :,2 ) = J( :,2 )/(obj.T + 273.15);
            %--------------------------------------------------------------
            % Derivative wrt Beta2
            %--------------------------------------------------------------
            J( :,3 ) = obj.SOC*W*exp(B1/(obj.T + 273.15) + B2*obj.SOC).*X.^Idx;
            %--------------------------------------------------------------
            % Derivative wrt Z
            %--------------------------------------------------------------
            J( :,4 ) = W*exp(B1/(obj.T + 273.15) + B2*obj.SOC).*log(X).*X.^Idx;
        end
        
        function  Yhat = predictions( obj, X, Beta )                                
            %--------------------------------------------------------------
            % Martinez-Laserna Model predictions
            %
            % Yhat = obj.predictions( X, Beta );
            %
            % Input Arguments:
            %
            % X       --> Independent data
            % Beta    --> Coefficient Vector {obj.Theta}. Assumed format:
            %             [W, B1, B2, Z].'
            %--------------------------------------------------------------
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            [W, B1, B2, Idx] = obj.assignPars( Beta );                      % Assign the parameters
            Yhat = W*exp( B1/(273.15+obj.T) + B2*obj.SOC )*X.^Idx;                                             % Predictions
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
            V(4) = B( 2 );                                                  % Initial value for power law index
            %--------------------------------------------------------------
            % Make the assumption that (Beta2*SOC + Beta1(T+273.15)) = 0. 
            % Then solve for Omega
            %--------------------------------------------------------------
            V(1) = 10^B(1);                                                 % Initial estimate for Omega
            %--------------------------------------------------------------
            % Initially set Beta1 = 1 and then Beta2 = -1/(SOC*(T+273.15));
            %--------------------------------------------------------------
            V(2) = 1.0;
            V(3) = -V(2)./( obj.SOC*( 273.15+obj.T ) );
        end
    end % constructor and ordinary methods
    
    methods ( Access = protected )
    end % protected methods
    
    methods ( Static = true )
        function [X, Y, W] = parseInputs( X, Y, W )
            %--------------------------------------------------------------
            % Remove zero or negative data
            %
            % [X, Y, W] = RegFit.mlm.parseInputs( X, Y, W );
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
        
        function [W, B1, B2, Z] = assignPars( Beta )
            %--------------------------------------------------------------
            % Assign the parameter vector contents
            %
            % [Omega, Beta1, Beta2, Z] = obj.assignPars( Beta );
            % [Omega, Beta1, Beta2, Z] = RegFit.mlm.assignPars( Beta );
            %--------------------------------------------------------------
            W = Beta(1);
            B1 = Beta(2);
            B2 = Beta(3);                                                   
            Z = Beta(4);
        end
    end % Static methods
end