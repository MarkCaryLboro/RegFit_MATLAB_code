classdef mrm < RegFit.fitModel
    % Relaxation model due to Meng, J., Stroe, D.I., Ricco, M., Luo, G.,...
    % Swierczynski, M. and Teodorescu, R., 2018. A novel multiple ...
    % correction approach for fast open circuit voltage prediction of ...
    % lithium-ion battery. IEEE Transactions on Energy Conversion,...
    % 34(2), pp.1115-1123.

    properties
        Theta       double                                                  % Model parameter vector
        LB          double                                                  % Lower bound constraint for parameters
        UB          double                                                  % Upper bound constraint for parameters
    end
    
    properties ( SetAccess = protected )
        ParNames    string      = [ "K_1", "K_2", "K_3" ]                   % Parameter Names
    end
    
    properties ( Constant = true )
        NumFitCoeff int8   = 3                                              % Number of parameters to estimate from the data
        ModelName   RegFit.fitModelType         = "mrm"                     % Model name
    end     
    
    methods
        function obj = mrm( ReEstObj )
           %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.mrm( ReEstObj );
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
            %             [k1, k2, k3].'
            %
            % Output Arguments:
            % J       --> Jacobean: [df_dk1, df_dk2, df_dk3]
            %--------------------------------------------------------------
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            X = X( : );                                                     % Make column vector
            [  K1, K2, ~ ] = obj.assignPars( Beta );                        % Retrieve parameters
            %--------------------------------------------------------------
            % Calculate Jacobean for the HRM model
            %--------------------------------------------------------------
            J = [ X.^K2, K1.*X.^K2.*log(X), ones( size( X ) ) ];
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
            %             [K1, K2, K3].'
            %--------------------------------------------------------------       
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            X = X( : );                                                     % Vec operator (column vector)
            [ K1, K2, K3 ] = obj.assignPars( Beta );                        % Assign the parameters
            Yhat = K3 + K1 * X.^( K2 );                                     % Compute the predictions
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
            K3 = 0.999*min( Y );
            G = log( Y - K3 );
            Z = [ ones( size( X ) ) log( X ) ];
            Q = Z \ G;
            K1 = exp( Q( 1 ) );
            K2 = Q( 2 );
            V = [ K1, K2, K3 ].';
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
    end % constructor and ordinary methods
    
    methods ( Static = true )
        function [X, W ] = processInputs( X, W )
            %--------------------------------------------------------------
            % Eliminate necessary aberrant points and corresponding weights
            %
            % [X, W ] = obj.processInputs( X, W );
            %
            % Input Arguments:
            %
            % X     --> Regressor vector
            % W     --> Weight vector
            %--------------------------------------------------------------
            P = ( X <= 0 );
            X = X( ~P );
            W = W( ~P );
        end       
        
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
            if ( nargin > 2 ) || ~isempty( W )
                W = W( ~P );
            end
        end % parseInputs
        
        function [K1, K2, K3] = assignPars( Beta )
            %--------------------------------------------------------------
            % Assign the parameter vector contents
            %
            % [K1, K2, K3] = obj.assignPars( Beta );
            % [K1, K2, K3] = RegFit.hrm.assignPars( Beta );
            %--------------------------------------------------------------
            K1 = Beta( 1 );
            K2 = Beta( 2 );                                                   
            K3 = Beta( 3 );
        end % assignPars
    end % static methods
end % mrm class