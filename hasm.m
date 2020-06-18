classdef hasm < RegFit.fitModel
    % Hypothesised active site model
    
    properties
        Theta       double                                                  % Model parameter vector
    end
    
    properties ( SetAccess = protected )
        ParNames    string = ["c_0", "c_{2,ref}", "m_1", "m_2", "m_3"]      % Parameter names
    end
    
    properties ( Constant = true )
        NumFitCoeff int8   = 5                                              % Number of parameters to estimate from the data
        LB          double = [-10 -10 -10 -10 -10].';                       % Parameter lower bounds
        UB          double = [10 10 10 10 10].';                            % Parameter upper bounds
        ModelName   RegFit.fitModelType = "hasm"                            % Model name
    end    
    
    methods
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
            %
            % Output Arguments:
            %
            % V     --> Initial estimate vector: [C0, C2,ref, m1, m2, m3].'
            %--------------------------------------------------------------
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
            %             [C0, C2,ref, m1, m2, m3].'
            %
            % Output Arguments:
            % J       --> Jacobean: [df_dC0, df_dC2, df_dm1, df_dm2, df_dm3]
            %----------------------------------------------------------------
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
        end        
    end % constructor and ordinary methods
    
    methods ( Static = true )
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
        
        function [W, B1, B2, Z] = assignPars( Beta )
            %--------------------------------------------------------------
            % Assign the parameter vector contents
            %
            % [Omega, Beta1, Beta2, Z] = obj.assignPars( Beta );
            % [Omega, Beta1, Beta2, Z] = RegFit.som.assignPars( Beta );
            %--------------------------------------------------------------
            W = Beta(1);
            B1 = Beta(2);
            B2 = Beta(3);                                                   
            Z = Beta(4);
        end
    end % Static methods    
end