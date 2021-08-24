classdef hrm < RegFit.fitModel
    % Hoster-Burrell (Lancaster) relaxation model
    
    properties 
        Theta       double                                                  % Model parameter vector
        LB          double                                                  % Lower bound constraint for parameters
        UB          double                                                  % Upper bound constraint for parameters
    end
    
    properties ( SetAccess = protected )
        ParNames    string      = [ "A_0", "A_1", "A_2", "A_3" ]            % Parameter Names
    end
    
    properties ( Constant = true )
        NumFitCoeff int8   = 4                                              % Number of parameters to estimate from the data
        ModelName   RegFit.fitModelType         = "hrm"                     % Model name
    end    
    
    properties ( Access = private, Dependent = true )
        NfitC_      double                                                  % Number of fit coefficients converted to double
    end 
    
    methods
        function obj = hrm( ReEstObj )
           %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.hrm( ReEstObj );
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
            %             [a0, a1, a2, a3].'
            %
            % Output Arguments:
            % J       --> Jacobean: [df_da0, df_da1, df_da2, df_da3]
            %--------------------------------------------------------------
            if ( nargin < 3 )
                Beta = obj.Theta;                                           % Apply default
            end
            X = X( : );                                                     % Make column vector
            [ ~, A1, A2, A3 ] = obj.assignPars( Beta );                     % Retrieve parameters
            %--------------------------------------------------------------
            % Calculate Jacobean for the HRM model
            %--------------------------------------------------------------
            J = [ ones( numel( X ), 1 ), log(1 - A2.*exp(-X./A3)),...
                (A1.*exp(-X./A3))./(A2.*exp(-X./A3) - 1),...
                (A1.*A2.*X.*exp(-X./A3))./(A3.^2.*(A2.*exp(-X./A3) - 1) ) ];
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
            [ A0, A1, A2, A3 ] = obj.assignPars( Beta );                    % Assign the parameters
            Yhat = A0 + A1 * log( 1 - A2 * exp( -X./A3 ) );                 % Compute the predictions
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
            A0 = 1.01 * max( Y );
            A1 = A0;
            G = log( 1 - exp( ( Y - A0 )/ A1 ) );
            Z = [ ones( size( X ) ) X ];
            Q = Z \ G;
            A2 = exp( Q( 1 ) );
            A3 = -1 / Q( 2 );
            V = [ A0, A1, A2, A3 ].';
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
        
        function [A0, A1, A2, A3] = assignPars( Beta )
            %--------------------------------------------------------------
            % Assign the parameter vector contents
            %
            % [A0, A1, A2, A3] = obj.assignPars( Beta );
            % [A0, A1, A2, A3] = RegFit.hrm.assignPars( Beta );
            %--------------------------------------------------------------
            A0 = Beta(1);
            A1 = Beta(2);
            A2 = Beta(3);                                                   
            A3 = Beta(4);
        end % assignPars
    end % static methods
end % hrm