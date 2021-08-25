classdef yrm < RegFit.fitModel
    % Yang, Wang, ..., Lei Relaxation Model
    %
    % Yang, J., Du, C., Wang, T., Gao, Y., Cheng, X., Zuo, P., Ma, Y., 
    % Wang, J., Yin, G., Xie, J. and Lei, B., 2018. 
    % Rapid prediction of the open-circuit-voltage of lithium ion batteries 
    % based on an effective voltage relaxation model. Energies, 11(12), 
    % p.3444.
    
    properties
        Theta       double                                                  % Model parameter vector
        LB          double                                                  % Lower bound constraint for parameters
        UB          double                                                  % Upper bound constraint for parameters
    end
    
    properties ( SetAccess = protected )
        ParNames    string      = [ "k_1", "k_2", "k_3", "k_4", "V_0" ]     % Parameter Names
    end    
    
    properties ( Constant = true )
        NumFitCoeff int8   = 5                                              % Number of parameters to estimate from the data
        ModelName   RegFit.fitModelType         = "yrm"                     % Model name
    end    
    
    properties ( Access = private, Dependent = true )
        NfitC_      double                                                  % Number of fit coefficients converted to double
    end    
    
    methods
        function obj = yrm( ReEstObj )
           %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.yrm( ReEstObj );
            %
            % Input Arguments:
            %
            % ReEstObj  --> RegFit.reEstLamda object. Implements
            %               regularisation parameter re-estimation 
            %               algorithm.
            %--------------------------------------------------------------
            arguments
                ReEstObj    (1,1)   
            end
            obj.ReEstObj = ReEstObj;
        end % constructor
 
        function J = jacobean( obj, X, Beta )                                        
            %--------------------------------------------------------------
            % Return Jacobean matrix
            %
            % J = obj.jacobean( X, Beta );
            %
            % Input Arguments:
            %
            % X       --> Dependent data
            % Beta    --> Coefficient Vector {obj.Theta}. Assumed format:
            %             [V0, k1, k2, k3, k4].'
            %
            % Output Arguments:
            % J       --> Jacobean: [df_dV0, df_dk1, df_dk2, df_dk3,...
            %                        df_dk4]
            %--------------------------------------------------------------
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            X = X( : );                                                     % Make column vector
            [ ~, K1, K2, K3, K4 ] = obj.assignPars( Beta );                 % Retrieve parameters
            %--------------------------------------------------------------
            % Calculate Jacobean for the YRM model
            %--------------------------------------------------------------
            J = [ ones( numel( X ), 1 ), -X.^K2, -K1.*X.^K2.*log(X),...
                  -X.^K4.*log(X), -K3.*X.^K4.*log(X).^2];
        end % jacobean
        
        function Yhat = predictions( obj, X, Beta )                                  
            %--------------------------------------------------------------
            % Hoster Relaxation Model predictions
            %
            % Yhat = obj.predictions( X, Beta );
            %
            % Input Arguments:
            %
            % X       --> Independent data
            % Beta    --> Coefficient Vector {obj.Theta}. Assumed format:
            %             [A0, A1, A2, A3].'
            %--------------------------------------------------------------       
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            X = X( : );                                                     % Vec operator (column vector)
            [ V0, K1, K2, K3, K4 ] = obj.assignPars( Beta );                % Assign the parameters
            Yhat = V0 - K1 .* X.^K2 - K3 .* X.^K4 .* log(X);
        end % predictions
        
        function V = startingValues( obj, X, Y )                            %#ok<INUSL>
            %--------------------------------------------------------------
            % Estimate starting parameter coefficient values
            %
            % V = obj.startingValues( X, Y );
            %
            % Input values:
            %
            % X     --> Input data
            % Y     --> Observed response data
            %--------------------------------------------------------------
            X = X( : );
            Y = Y( : );
            V0 = 0.99 * min( Y );
            K1 = 0;
            K2 = 1;
            G = log( ( V0 - Y ) ./ log( X ) );
            Z = [ ones( size( X ) ) log( X ) ];
            Q = Z \ G;
            K3 = exp( Q( 1 ) );
            K4 = Q( 2 );
            V = real( [ V0, K1, K2, K3, K4 ].' );
        end % startingValues
        
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
    end % Constructor and ordinary methods
    
    methods
        function N = get.NfitC_( obj )
            N = double( obj.NumFitCoeff );
        end
    end % set/get methods    
    
    methods ( Static = true )
        function [X, Y, W] = parseInputs( X, Y, W )
            %--------------------------------------------------------------
            % Remove negative data
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
        end % parseInputs
        
        function [V0, K1, K2, K3, K4] = assignPars( Beta )
            %--------------------------------------------------------------
            % Assign the parameter vector contents
            %
            % [V0, K1, K2, K3, K4] = obj.assignPars( Beta );
            % [V0, K1, K2, K3, K4] = RegFit.hrm.assignPars( Beta );
            %--------------------------------------------------------------
            V0 = Beta(1);
            K1 = Beta(2);
            K2 = Beta(3);                                                   
            K3 = Beta(4);
            K4 = Beta(5);
        end % assignPars    
    end % static methods
end % yrm

