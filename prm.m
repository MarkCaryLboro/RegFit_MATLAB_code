classdef prm < RegFit.fitModel
    % Pei et al relaxation model
    
    properties 
        Theta       double                                                  % Model parameter vector
        LB          double                                                  % Lower bound constraint for parameters
        UB          double                                                  % Upper bound constraint for parameters
    end
    
    properties ( SetAccess = protected )
        ParNames    string      = [ "OCV", "alpha", "beta" ]                % Parameter Names
    end
    
    properties ( Constant = true )
        NumFitCoeff int8   = 3                                              % Number of parameters to estimate from the data
        ModelName   RegFit.fitModelType         = "prm"                     % Model name
    end        
    
    methods
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
            %             [OCV, A, B].'
            %
            % Output Arguments:
            % J       --> Jacobean: [df_OCV, df_dA, df_dB]
            %--------------------------------------------------------------
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            X = X( : );                                                     % Make column vector
        end % jacobean
        
        function Yhat = predictions( obj, X, Beta )                                  
            %--------------------------------------------------------------
            % Pei et al Relaxation Model predictions
            %
            % Yhat = obj.predictions( X, Beta );
            %
            % Input Arguments:
            %
            % X       --> Independent data
            % Beta    --> Coefficient Vector {obj.Theta}. Assumed format:
            %             [OCV, A, B].'
            %--------------------------------------------------------------       
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            X = X( : );                                                     % Vec operator (column vector)
        end % predictions
        
        function V = startingValues( obj, X, Y )                            
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

        end % startingValues
        
        function B = basis( obj, X )
        end % basis
        
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
        function [X, Y, W] = parseInputs( X, Y, W )
            %--------------------------------------------------------------
            % Remove negative or zero data
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
        
        function [OCV, A, B] = assignPars( Beta )
            %--------------------------------------------------------------
            % Assign the parameter vector contents
            %
            % [OCV, A, B] = obj.assignPars( Beta );
            % [OCV, A, B] = RegFit.hrm.assignPars( Beta );
            %--------------------------------------------------------------
            OCV = Beta(1);
            A = Beta(2);
            B = Beta(3);                                                   
        end % assignPars
    end % static methods
end % prm