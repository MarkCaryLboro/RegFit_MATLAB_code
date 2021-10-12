classdef prm < RegFit.fitModel
    % Relaxation model due to Pei, L., Wang, T., Lu, R. and Zhu, C., 2014. 
    % Development of a voltage relaxation model for rapid open-circuit 
    % voltage prediction in lithium-ion batteries. Journal of Power Sources, 
    % 253, pp.412-418.
    
    properties 
        Theta       double                                                  % Model parameter vector
        LB          double                                                  % Lower bound constraint for parameters
        UB          double                                                  % Upper bound constraint for parameters
    end
    
    properties ( SetAccess = protected )
        Tct         (1,1)   double      { mustBePositive( Tct ), ...        % Charge transfer over potential time
                                          mustBeReal( Tct ) }    = (1/60)                                     
        ParNames    string      = [ "OCV", "alpha", "beta" ]                % Parameter Names
    end
    
    properties ( Constant = true )
        NumFitCoeff int8   = 3                                              % Number of parameters to estimate from the data
        ModelName   RegFit.fitModelType         = "prm"                     % Model name
    end        
    
    properties ( Access = private, Dependent = true )
    end % private and dependent properties
    
    methods
        function obj = prm( ReEstObj, Tct )
           %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.prm( ReEstObj, Tct );
            %
            % Input Arguments:
            %
            % ReEstObj  --> RegFit.reEstLamda object. Implements
            %               regularisation parameter re-estimation 
            %               algorithm.
            % Tct       --> 
            %--------------------------------------------------------------            
            obj = obj.setTct( Tct );
            obj.ReEstObj = ReEstObj;
        end % constructor
        
        function  [L, G, Lam] = costFcn( obj, Beta, X, Y, W, NumCovPar )
            %--------------------------------------------------------------
            % Regularised WLS cost function. Set weights to 1 for OLS.
            %
            % L = obj.costFcn( Beta, X, Y, W, NumCovPar );
            %
            % Input Arguments:
            %
            % Beta          --> Coefficient Vector {obj.Theta}. 
            % X             --> Independent data
            % Y             --> Observed data vector
            % W             --> Weights
            % NumCovPar     --> Number of covariance parameters
            %
            % Output Arguments:
            %
            % L             --> Value of the cost function
            % G             --> Analytical gradient of the cost function
            %                   with respect to the parameters
            % Lam           --> Updated Lamda value
            %--------------------------------------------------------------
            Res = obj.calcResiduals( X, Y, Beta );
            %--------------------------------------------------------------
            % Update the lamda value
            %--------------------------------------------------------------
            J = obj.jacobean( X, Beta );
            obj.ReEstObj = obj.ReEstObj.optimiseLamda( Res, W, J, NumCovPar );
            Lam = obj.ReEstObj.Lamda;
            %--------------------------------------------------------------
            % Regularised weighted least squares
            %--------------------------------------------------------------
            L = 0.5*sum( Res.^2./W ) + 0.5*Lam*(Beta.'*Beta);               % Regularised WLS cost function value
            G = obj.gradients( Beta, X, Y, W, Lam );                        % Analytical gradients 
        end % costFcn
                 
        function obj = mleRegTemplate( obj, X, Y, W, NumCovPar, Options )
            %--------------------------------------------------------------
            % Regularised MLE for model
            %
            % obj = obj.mleRegTemplate( X, Y, W, NumCovPar, Options );
            %
            % Input Arguments:
            %
            % X         --> Input data vector
            % Y         --> Observed response vector
            % W         --> Weight vector {1}
            % NumCovPar --> Number of covariance parameters
            % Options   --> Optimisation configuration object. Create with
            %               Options = optimoptions( 'fmincon' );
            %--------------------------------------------------------------
            if ( nargin <  4 ) || isempty( W )
                W = ones( size( X ) );
            end
            
            if ( nargin < 6)
                Options = optimoptions( 'fmincon' );
                Options.Display = 'Iter';
                Options.SpecifyObjectiveGradient = true;
            end
            %--------------------------------------------------------------
            % Generate starting values if required
            %--------------------------------------------------------------
            if isempty( obj.Theta )
                X0 = obj.startingValues( X, Y );
            else
                X0 = obj.Theta;
            end
            %--------------------------------------------------------------
            % Set up and execute regularised WLS PROBLEM
            %--------------------------------------------------------------
            C = obj.mleConstraints( X0 );
            PROBLEM = obj.setUpMLE( X0, X, Y, W, C, NumCovPar, Options );
            obj.Theta = fmincon( PROBLEM );
            [ ~, ~, Lam] = feval( PROBLEM.objective, obj.Theta);
            obj.ReEstObj = obj.ReEstObj.setLamda2Value( Lam );
            J = obj.jacobean( X, obj.Theta );
            Res = obj.calcResiduals( X, Y, obj.Theta );
            obj.ReEstObj = obj.ReEstObj.calcDoF( W, J, Lam );               % Effective number of parameters
            obj.ReEstObj = obj.ReEstObj.getMeasure( Lam, Res,...            % Calculate the performance measure
                W, J, NumCovPar );
        end % mleRegTemplate
            
        function obj = setTct( obj, Tct)
            %--------------------------------------------------------------
            % Set the charge transfer overpotential time to the preferred
            % time [h].
            %
            % obj = obj.setTct( Tct );
            %
            % Input Arguments:
            %
            % Tct   --> Charge transfer overpotential time [h] {1/60};
            %--------------------------------------------------------------
            obj.Tct = Tct;
        end % setTct
        
        function J = jacobean( obj, X, Beta )                                        
            %--------------------------------------------------------------
            % Return Jacobean matrix
            %
            % J = obj.jacobean( X, Beta );
            %
            % Input Arguments:
            %
            % X       --> (Nx2) Matrix of independent data. First column is
            %                   time, second is voltage
            % Beta    --> (3x1) Vector of coefficients in the order
            %                   [ OCV, alpha, beta ]
            %
            % Output Arguments:
            %
            % J       --> Jacobean: [df_OCV, df_dA, df_dB]
            %--------------------------------------------------------------
            if ( nargin < 3 ) || isempty( Beta )
                Beta = obj.Theta;
            else
                Beta = reshape( Beta, obj.NumFitCoeff, 1 );
            end
            [ OCV, A, B ] = obj.assignPars( Beta );
            [ U, T, DT ] = obj.getRegressors( X );
            T = T( 2:end );
            J = zeros( numel( T ), obj.NumFitCoeff );
            J( :, 1 ) = 1 - exp( -DT ./ ( A*T + B ) );
            H = exp( -DT  ./ ( A*T + B ) ) .* DT ./ ( A*T + B ).^2;
            J( :, 2 ) = ( U( 1:end - 1 ) - OCV ) .* T.* H;
            J( :, 3 ) = ( U( 1:end - 1 ) - OCV ) .* H;
        end % jacobean
        
        function Yhat = predictions( obj, X, Beta )                                  
            %--------------------------------------------------------------
            % Pei et al Relaxation Model predictions
            %
            % Yhat = obj.predictions( X, Beta );
            %
            % Input Arguments:
            %
            % X       --> (Nx2) Matrix of independent data. FIrst column is
            %                   time [h], second is voltage
            % Beta    --> Coefficient Vector {obj.Theta}. Assumed format:
            %             [OCV, A, B].'
            %--------------------------------------------------------------       
            if ( nargin < 3 ) || isempty( Beta )
                Beta = obj.Theta;                                           % Apply default
            end
            [ U, T, DT ] = obj.getRegressors( X );
            T = T( 2:end );
            [ OCV, A, B ] = obj.assignPars( Beta );
            H = exp( -DT ./ ( A*T + B ) );
            Yhat = U( 1:end-1 ) .* H + ( 1 - H ) .* OCV;
        end % predictions
        
        function V = startingValues( obj, X, ~ )                             
            %--------------------------------------------------------------
            % Estimate starting parameter coefficient values
            %
            % V = obj.startingValues( X, Y );
            %
            % Input values:
            %
            % X     --> (Nx1) Vector of independent data. 
            % Y     --> (Nx1) Vector of response data
            %
            % Output Arguments:
            % 
            % V     --> Vector of coefficients [OCV, A, B].'
            %--------------------------------------------------------------
            V = zeros( obj.NumFitCoeff, 1 );
            [ U, T, DT ] = obj.getRegressors( X );
            OCV = 1.1 * max( U );
            T = T( 2:end );
            Z = log( OCV - U( 2:end ) ) - log( OCV - U( 1:end - 1 ) );
            Z = -DT ./ Z;
            Idx = isfinite( Z );
            Z = Z( Idx );
            T = T( Idx );
            X = [ T, ones( numel( Z ), 1 ) ];
            V( 2:end ) = X\Z;
            V( 1 ) = OCV;
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
    end % Get/Set methods
    
    methods ( Access = protected )
    end % protected methods
    
    methods ( Access = private )     
        function PROBLEM = setUpMLE( obj, X0, X, Y, W, C, NumCovPar, Options )
            %--------------------------------------------------------------
            % Set up the MLE minimisation problem
            %
            % PROBLEM = obj.setUpMLE( X0, X, Y, W, C, NumCovPar, Options );
            %
            % Input Arguments:
            %
            % X0        --> Initial parameter estimates
            % X         --> Input data vector
            % Y         --> Observed response vector
            % W         --> Weight vector
            % C         --> Constraint structure.
            % NumCovPar --> Number of covariance parameters
            % Options   --> Optimisation configuration object. Create with
            %               Options = optimoptions( 'fmincon' );
            %--------------------------------------------------------------
            fh = @(Beta)obj.costFcn( Beta, X, Y, W, NumCovPar );
            PROBLEM.objective = fh;
            PROBLEM.x0 = X0;
            PROBLEM.Aineq = C.Aineq;
            PROBLEM.bineq = C.bineq;
            PROBLEM.Aeq = C.Aeq;
            PROBLEM.beq = C.beq;
            PROBLEM.lb = obj.LB;
            PROBLEM.ub = obj.UB;
            PROBLEM.nonlcon = C.nonlcon;
            PROBLEM.options = Options;
            PROBLEM.solver = 'fmincon';
        end
    end % private methods
    
    methods ( Static = true )
        function [ U, T, DT ] = getRegressors( X )
            %--------------------------------------------------------------
            % Fetch the regressor variables for the nonlinear equation.
            % Data for time <= obj.TcT is automatically set aside.
            %
            % [ U, T, DT ] = obj.getRegressors( X );
            %
            % Input Arguments:
            %
            % X       --> (Nx2) Matrix of independent data. FIrst column is
            %                   time [h], second is voltage
            %
            % Output Arguments:
            %
            % U     --> Relaxation voltage
            % T     --> Time [h]
            % DT    --> Delta time [h]
            %--------------------------------------------------------------
            T = X( :, 1 );
            DT = diff( T );
            U = X( :, 2 );
        end %getRegressors        
        
        function [X, Y, W] = parseInputs( X, Y, W, TcT )
            %--------------------------------------------------------------
            % Remove data for X > obj.TcT
            %
            % [X, Y, W] = RegFit.mlm.parseInputs( X, Y, W, TcT );
            % [X, Y, W] = obj.parseInputs( X, Y, W );
            %
            % X             --> Independent data
            % Y             --> Observed data vector
            % W             --> Weights
            % TcT           --> Threshold time
            %--------------------------------------------------------------
            P = ( X <= TcT ) | ( Y <= 0 );
            X = X( ~P );
            Y = Y( ~P );
            W = W( ~P );
            X = [ X, Y ];
            Y = Y( 2:end );
            W = W( 2:end );
        end % parseInputs

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
            % TcT   --> Threshold time
            %--------------------------------------------------------------
            P = any( X <= 0, 2 );
            X = X( ~P, : );
            W = W( ~P( 2:end ) );
        end        
                
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