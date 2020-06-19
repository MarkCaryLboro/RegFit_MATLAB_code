classdef fitModel
    % Nonlinear ridge regression 
    
    properties ( SetAccess = protected )
        ReEstObj   { mustBeReEstObj( ReEstObj ) }                           % Lamda re-estimation algorithm
    end
    
    properties ( SetAccess = protected, Dependent = true )
        Lamda       double                                                  % Regularisation parameter
    end
    
    properties ( Abstract = true )
        Theta       double                                                  % Model parameter vector
        LB          double                                                  % Lower bound constraint for parameters
        UB          double                                                  % Upper bound constraint for parameters
    end
    
    properties ( SetAccess = protected, Abstract = true )
        ParNames    string                                                  % Parameter Names
    end
    
    properties ( Constant = true, Abstract = true )
        ModelName   RegFit.fitModelType                                     % Model name
    end    
    
    methods ( Abstract = true )
        J = jacobean( obj, x, beta )                                        % Return Jacobean matrix
        Yhat = predictions( obj, x, beta )                                  % Model predictions
        obj = startingValues( obj, x, y )                                   % Starting estimates for nonlinear regression
    end % abstract method signatures
    
    methods ( Access = protected, Abstract = true )
    end % abstract and protected methods
    
    methods ( Static = true, Abstract = true )
        [X, Y, W] = parseInputs( X, Y, W )                                  % Pre-process data and weights
        varargout = assignPars( Beta )                                      % Assign parameters
    end % abstract static method signatures
    
    methods
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
        end
        
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
            % Parse the input data
            %--------------------------------------------------------------
            [X, Y, W] = obj.parseInputs( X, Y, W );
            %--------------------------------------------------------------
            % Generate starting values
            %--------------------------------------------------------------
            X0 = obj.startingValues( X, Y );
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
        end
        
        function [Ax, H] = diagnosticPlots( obj, X, Y, W )
            %--------------------------------------------------------------
            % Model diagnostic plots
            %
            % obj.diagnosticPlots( X, Y, Yhat, W );
            %
            % Input Arguments:
            %
            % X             --> Input data vector
            % Y             --> Observed response vector
            % Yhat          --> Predicted response vector
            % W             --> Weight vector
            %
            % Output Arguments:
            %
            % Ax            --> Axes handles
            % H             --> Line handles
            %--------------------------------------------------------------
            figure;
            Ax{ 1 } = subplot( 2, 2, 1 );
            H{ 1 } = obj.fitsPlot( Ax{ 1 }, X, Y );
            Ax{ 2 } = subplot( 2, 2, 2 );
            H{ 2 } = obj.weightedResPlot( Ax{ 2 }, X, Y, W );
            Ax{ 3 } = subplot( 2, 2, 3);
            H{ 3 } = obj.normalPlot( Ax{ 3 }, X, Y, W );
            Ax{ 4 } = subplot( 2, 2, 4 );
            H{ 4 } = obj.dataVsPredPlot( Ax{ 4 }, X, Y );
        end
        
        function [ Res, Yhat] = calcResiduals( obj, X, Y, Beta )
            %--------------------------------------------------------------
            % calculate the residuals
            %
            % [ Res, Yhat] = obj.calcResiduals( X, Y, Beta );
            %
            % Input Arguments:
            %
            % X             --> Input data vector
            % Y             --> Observed response vector
            % Beta          --> Coefficient Vector {obj.Theta}
            %
            % Output Arguments:
            %
            % Res   --> Ordinary residuals
            % Yhat  --> Model predictions
            %--------------------------------------------------------------
            if ( nargin < 4 )
                Beta = obj.Theta;                                           % Apply default
            end
            Yhat = obj.predictions( X, Beta );                              % Calculate the predictions
            Res = Y - Yhat;                                                 % Form the residuals
        end
        
        function obj = updateLamda( obj, X, Y, W, NumCovPar, MaxIter )
            %--------------------------------------------------------------
            % Update the regularisation coefficient
            %
            % obj = obj.updateLamda( X, Y, W, NumCovPar, MaxIter );
            %
            % Input Arguments:
            %
            % X         --> Input data vector
            % Y         --> Observed response vector
            % W         --> Weights
            % NumCovPar --> Number of covariance parameters
            % MaxIter   --> Maximum number of iterations for re-estimation
            %               algorithm {25}
            %--------------------------------------------------------------
            if ( nargin < 6 ) || isempty( MaxIter )
                MaxIter = 25;
            end
            %--------------------------------------------------------------
            % Parse the input data
            %--------------------------------------------------------------
            [X, Y, W] = obj.parseInputs( X, Y, W );
            %--------------------------------------------------------------
            % Re-estimate lamda based on the current fit coefficients and
            % weights
            %--------------------------------------------------------------
            Lam = obj.ReEstObj.Lamda;
            Res = obj.calcResiduals( X, Y );
            J = obj.jacobean( X );
            obj.ReEstObj = obj.ReEstObj.optimiseLamda( Lam, Res, W, J,...
                NumCovPar, MaxIter );
        end
        
        function SE = stdErrors( obj, X, W, Beta )
            %--------------------------------------------------------------
            % Calculate the standard errors for the parameter estimates
            %
            % SE = obj.stdErrors( X, W, Beta );
            %
            % Input Arguments:
            %
            % X         --> Input data vector
            % W         --> Data weights
            % Beta      --> Coefficient Vector {obj.Theta}
            %--------------------------------------------------------------
            if ( nargin < 4 )
                Beta = obj.Theta;
            end
            P = ( X <= 0 );
            X = X( ~P );
            W = W( ~P );
            J = obj.jacobean( X, Beta );
            [ ~, Z ] = obj.ReEstObj.calcSmatrix( obj.Lamda, W, J );
            H = ( Z.'*Z + obj.Lamda*eye( obj.NumFitCoeff ) )\eye( obj.NumFitCoeff );
            SE = sqrt( diag( H*( Z.'*Z )*H ) );
        end
        
        function [ParameterVectors, Ave, StanDev] = bootStrapSamples( obj, X, Y, W, NumCovPar, N )
            %--------------------------------------------------------------
            % Perform the bootstrap for the number of samples specified
            %
            % ParameterVectors = obj.bootStrapSamples( X, Y, W,...
            %                           NumCovPar, N );
            %
            % Input Arguments:
            %
            % X         --> Input data vector
            % Y         --> Observed response vector
            % W         --> Weights
            % NumCovPar --> Number of covariance parameters
            % N         --> Number of bootstrap samples {250}
            %
            %
            % Output Arguments:
            %
            % ParameterVectors  --> Parameter vectors for each bootstrap
            %                       sample ( obj.NumFitCoeff x N).
            % Ave               --> Average parameter vector from samples
            % StanDev           --> Standard deviation of samples
            %--------------------------------------------------------------
            if ( nargin < 6 )
                N = 250;                                                    % Apply default
            end
            %--------------------------------------------------------------
            % Define optimisation options
            %--------------------------------------------------------------
            Options = optimoptions( 'fmincon' );
            Options.Display = 'None';
            Options.SpecifyObjectiveGradient = true;
            NumPoints = numel( X );
            BootSamples = randi( NumPoints, NumPoints, N );
            %--------------------------------------------------------------
            % Model the boot strap data
            %--------------------------------------------------------------
            warning off all;
            ParameterVectors = nan( obj.NumFitCoeff + 1, N );
            fprintf('\n=======================================================');
            fprintf('\n             GENERATING BOOT STRAP SAMPLES             ');
            fprintf('\n=======================================================');
            for Q = 1:N
                fprintf('\n           BOOT STRAP SAMPLE #%5.0d           ', Q);
                bs_obj = obj;
                bs_obj.Lamda = 0.01;
                %----------------------------------------------------------
                % Perform the bootstrap
                %----------------------------------------------------------
                Xboot = X( BootSamples(:, Q) );
                Yboot = Y( BootSamples(:, Q) );
                Wboot = W( BootSamples(:, Q) );
                try
                    bs_obj = bs_obj.mleRegTemplate( Xboot, Yboot, Wboot, NumCovPar, Options );
                    %------------------------------------------------------
                    % Report parameters
                    %------------------------------------------------------
                    ParameterVectors( :, Q ) = [bs_obj.Theta; bs_obj.Lamda];
                catch
                    fprintf('\n  BOOT STRAP SAMPLE #%5.0d NOT ESTIMATED  ', Q);
                end
            end
            warning on all;
            %--------------------------------------------------------------
            % Remove any nans
            %--------------------------------------------------------------
            ParameterVectors = ParameterVectors( ~isnan( ParameterVectors ));
            P = numel( ParameterVectors );
            ParameterVectors = reshape( ParameterVectors, ( obj.NumFitCoeff + 1 ), P/double( obj.NumFitCoeff + 1) );
            fprintf('\n\n');
            %--------------------------------------------------------------
            % Generate statistics
            %--------------------------------------------------------------
            Ave = mean( ParameterVectors, 2 ).';
            StanDev = std( ParameterVectors, [], 2 ).';
            %--------------------------------------------------------------
            % Generate histograms
            %--------------------------------------------------------------
            obj.histogram( ParameterVectors, 25 );
        end
    end % constructor and ordinary methods
    
    methods
        function L = get.Lamda( obj )
            % Return regularisation parameter
            L = obj.ReEstObj.Lamda;
        end
        
        function obj = set.Lamda( obj, Value )
            % Set lamda to the desired value
            obj.ReEstObj = obj.ReEstObj.setLamda2Value( Value );
        end
    end % get/set methods
    
    methods ( Access = protected )     
        function C = mleConstraints( obj, Beta )                            %#ok<INUSD>
            %--------------------------------------------------------------
            % Provide custom constraints for optimisation. See help for
            % fmincon for definitions.
            %
            % Input Arguments:
            %
            % Beta  --> Coefficient vector. Decision variables for
            %           optimisation of the cost function for RIGLS
            %
            % Output Arguments:
            %
            % C     --> Structure of constraints with fields:
            %           Aineq       --> Linear inequality constraint
            %                           coefficient matrix.
            %           bineq       --> Linear inequality constraints bound
            %                           matrix.
            %           Aeq         --> Linear equality constraint
            %                           coefficient matrix.
            %           beq         --> Linear equality constraints bound
            %                           matrix.
            %           nonlcon     --> Nonlinear constraints function
            %--------------------------------------------------------------
            C.Aineq = [];
            C.bineq = [];
            C.Aeq = [];
            C.beq = [];
            C.nonlcon = [];
        end
        
        function G = gradients( obj, Beta, X, Y, W, Lam )
            %--------------------------------------------------------------
            % Gradient of the cost function with resepct to Beta
            %
            % G = obj.gradients( Beta, X, Y, W, Lam  )     
            %
            % Input Arguments:
            %
            % Beta  --> Coefficient Vector {obj.Theta}. Assumed 
            %           format: [Alpha, Beta, Eta, Z].'
            % X     --> Independent data
            % Y     --> Observed data vector
            % W     --> Weights
            % Lam   --> Regularisation coefficient
            %--------------------------------------------------------------
            J = obj.jacobean( X, Beta );
            F = obj.predictions( X, Beta );
            G = J.'*diag( (1./W) )*( ( F - Y ) ) + Lam*Beta;
        end
       end % protected methods
    
    methods ( Access = private )
        function histogram( obj, V, Bins )
            %--------------------------------------------------------------
            % Create histogram of parameter estimates
            %
            % obj.histogram( V, Bins );
            %
            % Input Arguments:
            %
            % V     --> Matrix of parameters arrannged by rows
            % Bins  --> Number of histogram bins {21}
            %--------------------------------------------------------------
            if ( nargin < 3 )
                Bins = 21;                                                  % Apply default
            end
            %--------------------------------------------------------------
            % Determine subplot layout
            %--------------------------------------------------------------
            NumCoeff = double( obj.NumFitCoeff );
            NumPlots = floor( NumCoeff/2 );
            if rem( NumCoeff, 2 ) == 1
                NumPlots = NumPlots + 1;
            end
            %--------------------------------------------------------------
            % Draw histograms
            %--------------------------------------------------------------
            figure;
            for Q = 1:NumCoeff
                subplot( NumPlots, 2, Q );
                histogram( V( Q,: ), Bins );
                grid on;
                ylabel( "Frequency [#]" );
                xlabel( sprintf('%s', obj.ParNames( Q )) );
            end
            %--------------------------------------------------------------
            % Draw histogram for Lamda
            %--------------------------------------------------------------
            figure;
            histogram( V( end,: ), Bins );
            grid on;
            ylabel( "Frequency [#]" );
            xlabel( '\lambda' );
        end
        
        function H = dataVsPredPlot( obj, Ax, X, Y )
            %--------------------------------------------------------------
            % Data versus predictions plot diagnostic
            %
            % H = obj.dataVsPredPlot( Ax, X, Y );
            %
            % Input Arguments:
            %
            % Ax        --> Axes handle
            % X         --> Input data vector
            % Y         --> Observed response vector
            %
            % Output Arguments:
            %
            % H         --> Handle to line objects 
            %--------------------------------------------------------------
            Yhat = obj.predictions( X );
            H = plot( Yhat, Y, 'bo' );
            H.MarkerFaceColor = 'blue';
            V = axis( Ax );
            Mx = max( V );
            Mn = min( V );
            axis( repmat( [Mn Mx], 1, 2 ) );
            hold( Ax, 'on' );
            H(2) = plot( Ax.XLim, Ax.YLim, 'b-' );
            axis( Ax, 'equal' );
            hold( Ax, 'off' );
            H(2).LineWidth = 2.0;
            grid on
            xlabel('Prediction');
            ylabel('Data');
            title('Data Versus Predicted');
        end
        
        function H = normalPlot( obj, Ax, X, Y, W ) 
            %--------------------------------------------------------------
            % Normal probability plot for weighted residuals
            %
            % H = obj.normalPlot( Ax, X, Y, W );
            %
            % Input Arguments:
            %
            % Ax        --> Axes handle
            % X         --> Input data vector
            % Y         --> Observed response vector
            % W         --> Weights
            %
            % Output Arguments:
            %
            % H         --> Handle to line objects 
            %--------------------------------------------------------------
            Res = obj.calcResiduals( X, Y );
            Res = Res./sqrt( W );
            H = normplot( Ax, Res );
        end
        
        function H = weightedResPlot( obj, Ax, X, Y, W ) 
            %--------------------------------------------------------------
            % Weighted residual versus predicted plot
            %
            % H = obj.weightedResPlot( Ax, X, Y, W );
            %
            % Input Arguments:
            %
            % Ax        --> Axes handle
            % X         --> Input data vector
            % Y         --> Observed response vector
            % W         --> Weights
            %
            % Output Arguments:
            %
            % H         --> Handle to line objects 
            %--------------------------------------------------------------
            [Res, Yhat] = obj.calcResiduals( X, Y );
            Res = Res./sqrt( W );
            H = plot( Ax, Yhat, Res, 'bo' );
            H.MarkerFaceColor = 'blue';
            grid on;
            xlabel('Prediction');
            ylabel('Weighted Residual');
            title('Weighted Residual Vs. Prediction');
        end
        
        function H = fitsPlot( obj, Ax, X, Y, NumPts)
            %--------------------------------------------------------------
            % Model fits to data
            %
            % H = obj.fitsPlot( Ax, X, Y, NumPts );
            %
            % Input Arguments:
            %
            % Ax        --> Axes handle
            % X         --> Input data vector
            % Y         --> Observed response vector
            % NumPts    --> Number of hi res points for fitted line {101}
            %
            % Output Arguments:
            %
            % H         --> Handle to line objects 
            %--------------------------------------------------------------
            if ( nargin < 5 ) || ( NumPts < numel( X ) )
                NumPts = 101;
            end
            Xhi = linspace( min( X ), max( X ), NumPts ).';
            Yhi = obj.predictions( Xhi );
            H = plot( Ax, X, Y, 'bo', Xhi, Yhi, 'r-' );
            H(1).MarkerFaceColor = 'blue';
            H(2).LineWidth = 2.0;
            grid on
            xlabel('X');
            ylabel('Y');
            title('Model Fits')
        end
        
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
    end % private and helper methods
    
    methods ( Hidden = true )
        
        function obj = olsRegTemplate( obj, X, Y, NumCovPar, Options )
            %--------------------------------------------------------------
            % Regularised MLE for model
            %
            % obj = obj.olsRegTemplate( X, Y, NumCovPar, Options );
            %
            % Input Arguments:
            %
            % X         --> Input data vector
            % Y         --> Observed response vector
            % NumCovPar --> Number of covariance parameters
            % Options   --> Optimisation configuration object. Create with
            %               Options = optimoptions( 'fmincon' );
            %--------------------------------------------------------------
            if ( nargin < 5)
                Options = optimoptions( 'fmincon' );
                Options.Display = 'Iter';
                Options.SpecifyObjectiveGradient = true;
            end
            %--------------------------------------------------------------
            % Parse the input data
            %--------------------------------------------------------------
            [X, Y, W] = obj.parseInputs( X, Y, ones( numel( X ), 1 ) );
            %--------------------------------------------------------------
            % Generate starting values
            %--------------------------------------------------------------
            X0 = obj.startingValues( X, Y );
            %--------------------------------------------------------------
            % Set up and execute regularised WLS PROBLEM
            %--------------------------------------------------------------
            PROBLEM = obj.setUpMLE( X0, X, Y, W, NumCovPar, Options );
            obj.Theta = fmincon( PROBLEM );
            Res = obj.calcResiduals( X, Y );
            J = obj.jacobean( X );
            obj.ReEstObj = obj.ReEstObj.optimiseLamda( obj.ReEstObj.Lamda,...
                Res, W, J, NumCovPar );
        end
    end
end

function mustBeReEstObj( ModelObj )
    %----------------------------------------------------------------------
    % Validator function for ModelObj property
    %
    % mustBeReEstObj( ModelObj )
    %----------------------------------------------------------------------
    if ~isempty( ModelObj ) && ~isa( ModelObj.Name,'RegFit.reEstType' )
        error('Unrecognised re-estimation model option');
    end
end