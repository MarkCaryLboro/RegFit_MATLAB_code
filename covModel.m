classdef covModel
    % covariance model for heteroscedastic dependency    
    
    properties ( Abstract = true )
        Delta   double                                                      % Heteroscedastic covariance model parameters
    end
    
    properties ( Constant = true, Abstract = true )
        LB      double                                                      % Lower bound constraint for parameters
        UB      double                                                      % Upper bound constraint for parameters
    end
    
    properties ( SetAccess = protected, Dependent = true )
        NumCoVParam                                                         % Number of covariance parameters
        Sigma                                                               % Standard deviation scale factor
    end
    
    properties ( SetAccess = protected )
        Sigma2  double                                                      % Variance scale parameter
    end
    
    methods ( Abstract = true )
        W = calcWeights( obj, Yhat )
    end % abstract method signatures
    
    methods ( Access = protected, Abstract = true )
    end % protected & abstract method signatures
    
    methods
        function obj = mleTemplate( obj, Ydata, Yhat )
            %--------------------------------------------------------------
            % Profile likelihood template
            %
            % obj = obj.mleTemplate( Ydata, Yhat );
            %
            % Input Arguments:
            %
            % Ydata --> Observed data
            % Yhat  --> Corresponding model predictions
            %--------------------------------------------------------------
            [Ydata, Yhat] = obj.parseData( Ydata, Yhat ); 
            Res = obj.calcResiduals( Ydata, Yhat );
            % Set up the optimisation problem
            PROBLEM.x0 = obj.Delta;
            PROBLEM.objective = @(D)obj.costFcn( D, Res, Yhat );
            PROBLEM.Aineq = [];
            PROBLEM.bineq = [];
            PROBLEM.Aeq = [];
            PROBLEM.beq = [];
            PROBLEM.lb = obj.LB;
            PROBLEM.ub = obj.UB;
            PROBLEM.nonlcon = [];
            options = optimoptions('fmincon');
            options.Display = 'None';
            options.OptimalityTolerance = 1e-3;
            options.Algorithm = 'interior-point';
            PROBLEM.options = options;
            PROBLEM.solver = 'fmincon';
            try
                obj.Delta = fmincon( PROBLEM );
            catch
                obj.Delta = nan;
            end
            [~, obj.Sigma2] = feval( PROBLEM.objective, obj.Delta);
        end
        
        function plotProfileLikelihood( obj, Ydata, Yhat, NumPts )
            %--------------------------------------------------------------
            % Plot the profile likelihood between the limits specified
            %
            % obj.plotProfileLikelihood( Ydata, Yhat, NumPts )
            %
            % Input Arguments:
            %
            % Ydata     --> Observed data
            % Yhat      --> Corresponding model predictions
            % NumPts    --> Number of samples for each parameter {101}
            %--------------------------------------------------------------
            if nargin<4
                NumPts = 101;                                               % Apply default
            end
            Res = obj.calcResiduals( Ydata, Yhat );
            %--------------------------------------------------------------
            % Create parameter sample
            %--------------------------------------------------------------
            switch obj.NumCoVParam
                case 1
                    % Power or exponential model
                    Par = obj.sampleParOne( NumPts );
                    obj.plotLikelihoodFcn( Par, Res, Yhat );
                case 2
                    % Two-components of variance model
                    Par = obj.sampleParTwo( NumPts );
                    obj.plotLikelihoodSurf( Par, Res, Yhat );
                otherwise
                    error('[Class]: RegFit.covModel, [Method]: plotProfileLikelihood,: [Msg]: "Unsupported number of covariance parameters"');
            end
        end
    end % ordinary methods
    
    methods
        % SET/GET METHODS
        function N = get.NumCoVParam( obj )
            % Return number of covariance model parameters
            N = numel( obj.Delta );
        end
        
        function S = get.Sigma( obj )
            % Return the standard error scale factor
            S = sqrt( obj.Sigma2 );
        end
    end % set/get methods
    
    methods ( Access = protected )
       function [L, Sigma2] = costFcn( obj, D, Res, Yhat )
            %--------------------------------------------------------------
            % Profile likelihood cost function
            %
            % L = obj.costFcn( D, Res, Yhat );
            %
            % Input Arguments:
            %
            % D         --> Power law exponent
            % Res       --> Residual vector
            % Yhat      --> Corresponding predictions
            %
            % Output Arguments:
            %
            % L         --> Profile likelihood
            % Sigma2    --> Variance scale factor
            %--------------------------------------------------------------
            obj.Delta = D;
            W = obj.calcWeights( Yhat );
            N = numel( W );
            Sigma2 = sum( (Res.^2)./W )/N;
            L = 0.5*N*log( Sigma2 ) + 0.5*sum( log( W ) ) + 0.5*N;
       end        
    end % protected methods
    
    methods ( Access = private )
        function P = sampleParOne( obj, NumPts )
            %--------------------------------------------------------------
            % Generate a set of parameters to support plotting - one factor
            %
            % P = obj.sampleParOne( NumPts );
            %
            % NumPts    --> Number of samples for  parameter
            %--------------------------------------------------------------
            if rem( NumPts, 2 ) == 0
                NumPts = NumPts + 1;                                        % Number of samples must be odd
            end
            P = linspace( obj.LB, obj.UB, NumPts ).';
        end
        
        function P = sampleParTwo( obj, NumPts )
            %--------------------------------------------------------------
            % Generate a set of parameters to support plotting - two factor
            %
            % P = obj.sampleParOne( NumPts );
            %
            % NumPts    --> Number of samples for each parameter
            %--------------------------------------------------------------
            if rem( NumPts, 2 ) == 0
                NumPts = NumPts + 1;                                        % Number of samples must be odd
            end
            X = linspace( obj.LB(1), obj.UB(1), NumPts );
            Y = linspace( obj.LB(2), obj.UB(2), NumPts ).';
            [X, Y] = meshgrid( X, Y);
            P = [X(:), Y(:)];
        end
        
        function plotLikelihoodFcn( obj, Par, Res, Yhat )
            %--------------------------------------------------------------
            % Plot the likelihood between the lower and upper bounds for
            % the parameter
            %
            % obj.plotLikelihoodFcn( Par, Res, Yhat );
            %
            % Input Arguments:
            %
            % Par   --> List of parameter values
            % Res   --> Residual vector
            % Yhat  --> Vector of predictions
            %--------------------------------------------------------------
            Lp = nan( size( Par ) );
            for Q = 1:numel( Par )
                Lp( Q ) = obj.costFcn( Par( Q ), Res, Yhat );
            end
            figure;
            plot( Par, Lp, 'b-', 'LineWidth', 2 );
            grid on
            xlabel('\theta');
            ylabel('L_p');
            title( sprintf( '%s', obj.CovName ) );
        end
        
        function plotLikelihoodSurf( obj, Par, Res, Yhat )
            %--------------------------------------------------------------
            % Plot the likelihood between the lower and upper bounds for
            % the parameter
            %
            % obj.plotLikelihoodSurf( Par, Res, Yhat );
            %
            % Input Arguments:
            %
            % Par   --> List of parameter values
            % Res   --> Residual vector
            % Yhat  --> Vector of predictions
            %-------------------------------------------------------------- 
            N = max( size( Par ) );
            Lp = nan( N, 1 );
            for Q = 1:N
                Lp( Q ) = obj.costFcn( Par( Q, : ), Res, Yhat );
            end
            figure;
            N = sqrt( N );
            X = reshape( Par(:, 1), N, N );
            Y = reshape( Par(:, 2), N, N );
            Lp = reshape( Lp, N, N );
            mesh( X, Y, Lp);
            grid on
            xlabel('\theta_1');
            ylabel('\theta_2');
            zlabel('L_p');
            title( sprintf( '%s', obj.CovName ) );
        end
    end % private methods
    
    methods ( Static = true )
        function Res = calcResiduals( Ydata, Yhat )
            %----------------------------------------------------------------------
            % Calculate the residuals
            %
            % Res = RegFit.calcResiduals( Ydata, Yhat );
            % Res = obj.calcResiduals( Ydata, Yhat );
            %
            % Input Arguments:
            %
            % Ydata --> Observed data
            % Yhat  --> Corresponding fit model predictions
            %----------------------------------------------------------------------
            Res = Ydata - Yhat;
        end
        
        function [Ydata, Yhat] = parseData( Ydata, Yhat )
            %--------------------------------------------------------------
            % remove points that have zero prediction and therefore
            % infinite weight
            %
            % [Ydata, Yhat] = RegFit.covModel( Ydata, Yhat);
            % [Ydata, Yhat] = obj.covModel( Ydata, Yhat);
            %
            % Input Arguments:
            %
            % Ydata --> Observed data
            % Yhat  --> Corresponding predictions
            %--------------------------------------------------------------
            P = ( Yhat <= 0 );
            Ydata = Ydata( ~P );
            Yhat = Yhat( ~P );
        end
    end % Static methods
end

