classdef regNonlinIGLS
    % Nonlinear regularised Iterative Generalied Least Squares
    
    properties ( SetAccess = immutable )
        X                   double                                          % Regressor vector
        Y                   double                                          % Observed data vector
        XLB(1, 2)           double                                          % Lower bound for regressor data [natural, coded] units
        XUB(1, 2)           double                                          % Upper bound for regressor data [natural, coded] units
        YLB(1, 2)           double                                          % Lower bound for observed data [natural, coded] units
        YUB(1, 2)           double                                          % Upper bound for observed data [natural, coded] units
        Xname               string                                          % Name of input data
        Yname               string                                          % Name of response data
    end
    
    properties ( SetAccess = immutable )
        UserName = getenv("username")                                       % Username for creator of class
        ComputerName = getenv("computername")                               % Computer name  
        Created = datetime( 'now' )                                         % Date and time object was created
    end
    
    properties ( SetAccess = protected )
        FitModelContextObj  { mustBeFitModelObj( FitModelContextObj ) }     % Regularised fit model context object
        CovModelContextObj  RegFit.covModelContext                          % Covariance model context object
    end
    
    properties ( SetAccess = protected, Dependent = true )
        Lamda               double                                          % Regularisation coefficient
        DoF                 double                                          % Model degree of freedom including covariance parameters
        Algorithm                                                           % Re-estimation algorithm name
        ModelName                                                           % Fit model type
        N                                                                   % Number of data points
        ParNames            string                                          % Parameter names
        Theta               double                                          % Fit parameter vector
        Xc                  double                                          % Coded regressor data
        Yc                  double                                          % Coded observed data
        W                   double                                          % Data weights for IGLS analysis
        Delta               double                                          % Covariance model parameters
        CovModel                                                            % Covariance model name
        Sigma2              double                                          % Variance scale parameter                                          
        Sigma               double                                          % Standard error scale parameter
        NumCovPar           double                                          % Number of covariance model parameters
        TotalNumPar         double                                          % Total number of model parameters
        Measure             double                                          % Model theoretic measure (AICc or BIC)
    end
    
    methods
        function obj = regNonlinIGLS( X, Y, fitModelObj, covModelObj )
            %--------------------------------------------------------------
            % Class constructor for nonlinear IGLS fitting 
            %
            % obj = RegFit.regNonlinIGLS( X, Y, fitModelObj, covModelObj );
            %
            % Input Arguments:
            %
            % X             --> Input data (regressor) structure.
            %                   Must have fields:
            %                       Data --> Vector of input data
            %                       LB   --> Mapping a --> ac for coding
            %                       UB   --> Mapping b --> bc for coding
            % Y             --> Observed response structure. Must have
            %                   fields:
            %                       Data --> Vector of input data
            %                       LB   --> Mapping a --> ac for coding
            %                       UB   --> Mapping b --> bc for coding
            % fitModelObj   --> RegFit.fitModel object
            % covModelObj   --> RegFit.covModel object
            %--------------------------------------------------------------
            obj.X = X.Data;
            obj.XLB = X.LB;
            obj.XUB = X.UB;
            try
                obj.Xname = X.Name;
            catch
                obj.Xname = "X";
            end
            obj.Y = Y.Data;
            obj.YLB = Y.LB;
            obj.YUB = Y.UB;
            try
                obj.Yname = Y.Name;
            catch
                obj.Yname = "Y";
            end
            obj.FitModelContextObj = RegFit.fitModelContext( fitModelObj, obj.Xc, obj.Yc );
            obj.CovModelContextObj = RegFit.covModelContext( covModelObj, obj.Yc );
        end
        
        function obj = regOLSestimates( obj, Options )
            %--------------------------------------------------------------
            % Regularised ordinary least squares fit
            %
            % obj = obj.regOLSestimates( Options );
            %
            % Input Arguments:
            %
            % Options   --> Optimisation configuration object. Create with
            %               Options = optimoptions( 'fmincon' );
            %               Automatically generated if not supplied
            %--------------------------------------------------------------
            if ( nargin < 2 ) || ~isa( Options, 'optim.options.Fmincon' )
                Options = optimoptions( 'fmincon' );
                Options.Display = 'Iter';
            end
            Options.SpecifyObjectiveGradient = true;
            %-------------------------------------------------------------
            % Weights must be one
            %--------------------------------------------------------------
            obj.FitModelContextObj = obj.FitModelContextObj.setWeights( ones( obj.N, 1) );         
            %--------------------------------------------------------------
            % ROLS fit
            %--------------------------------------------------------------
            obj.FitModelContextObj = obj.FitModelContextObj.nonLinRegFit(...
                obj.CovModelContextObj.NumCoVParam, Options );
            %--------------------------------------------------------------
            % Calculate predictions
            %--------------------------------------------------------------
            [ ~, YhatCoded ] = obj.predictions();
            obj.CovModelContextObj = obj.CovModelContextObj.setPredictions( YhatCoded );
        end
        
        function obj = regIGLS( obj, MaxIter, Options)
            %--------------------------------------------------------------
            % Regularised Iterative Least Squares algorithm
            %
            % obj = obj.regIGLS( MaxIter, Options );
            %
            % Input Arguments:
            %
            % MaxIter   --> Maximum number of iterations {5}
            % Options   --> Optimisation options. Use to override defaults
            %--------------------------------------------------------------
            warning off;
            if ( nargin < 2 ) || isempty( MaxIter )
                MaxIter = 5;                                                % Apply default
            end
            if ( nargin < 3 ) || ~isa( Options, 'optim.options.Fmincon' )
                Options = optimoptions( 'fmincon' );                        % Apply default
                Options.Display = 'None';
            end
            Options.SpecifyObjectiveGradient = true;
            %--------------------------------------------------------------
            % Initialise with OLS estimates
            %--------------------------------------------------------------
            obj = obj.regOLSestimates( Options );
            %--------------------------------------------------------------
            % Regularised IGLS
            %--------------------------------------------------------------
            Iter = 0;
            Stopflg = false;
            fprintf('\n\n==================================================');
            fprintf('\n|               IGLS Algorithm                   |');
            fprintf('\n==================================================\n');
            while ~Stopflg
                ThetaLast = obj.Theta;
                Iter = Iter + 1;
                fprintf( '\nIGLS Iteration #%d\n', Iter );
                obj.CovModelContextObj = obj.CovModelContextObj.profileLikelihood();
                Weights = obj.CovModelContextObj.calcWeights();
                obj.FitModelContextObj = obj.FitModelContextObj.setWeights( Weights );
                obj.FitModelContextObj = obj.FitModelContextObj.nonLinRegFit( obj.NumCovPar, Options );
                ConvFlg = 100*( norm(obj.Theta - ThetaLast )/norm( obj.Theta ) ) <= 0.0001;
                Stopflg = ConvFlg | ( Iter >= MaxIter );
            end
            warning on;
        end
        
                
        function J = jacobean( obj, X )   
            %----------------------------------------------------------------
            % Return Jacobean matrix
            %
            % J = obj.jacobean( X );
            %
            % Input Arguments:
            %
            % X       --> Dependent data {obj.X}
            %
            % Output Arguments:
            % J       --> Jacobean matrix
            %----------------------------------------------------------------
            if ( nargin < 2 )
                X = obj.X;                                                  % Apply default
            end 
            X = X(:);
            C = obj.codeX( X );
            J = obj.FitModelContextObj.jacobean( C );
        end
        
        function [ Yhat, YhatC ] = predictions( obj, X )
            %--------------------------------------------------------------
            % Return model predictions
            %
            % [ Yhat, YhatC ] = obj.predictions( X );
            %
            % Input Arguments:
            %
            % X     --> Regressor vector in natural units {obj.X}
            %
            % Output Arguments:
            %
            % Yhat  --> Predictions in natural units
            % YhatC --> Predictions in coded units
            %--------------------------------------------------------------
            if ( nargin < 2 )
                X = obj.X;
            end
            X = X(:);
            C = obj.codeX( X );
            YhatC = obj.FitModelContextObj.predictions( C );
            Yhat = obj.decodeY( YhatC );
        end
        
        function SE = stdErrors( obj )
            %--------------------------------------------------------------
            % Return standard errors for the fit parameters
            %
            % SE = obj.stdErrors();
            %
            %--------------------------------------------------------------
            SE = obj.FitModelContextObj.stdErrors();
        end
        
        function [LCI, UCI] = confInt( obj, P )
            %--------------------------------------------------------------
            % Confidence intervals at the specified p-level
            %
            % [ LCI, UCI ] = obj.confInt( P );
            %
            % Note confidence intervals are specified as ( 1 - P ), so for 
            % a 95% confidence interval P is 0.05.
            %
            % Input Arguments:
            %
            % P     --> alpha probability {0.05}
            %
            % Output Arguments:
            %
            % LCI   --> Lower confidence bound for parameter estimates
            % UCI   --> Upper confidence bound for parameter estimates
            %--------------------------------------------------------------
            if ( nargin < 2 ) || isempty( P )
                P = 0.05;
            end
            DF = obj.N - obj.TotalNumPar;
            SE = obj.Sigma*obj.stdErrors();
            T = abs(tinv( 0.5*P, DF ));                                     % T-statistic
            LCI = obj.Theta - SE*T;                                         % Lower C.I.
            UCI = obj.Theta + SE*T;                                         % Upper C.I.
        end
        
        function diagnosticPlots( obj )
            %--------------------------------------------------------------
            % Model diagnostic plots
            %
            % obj.diagnosticPlots();
            %
            %--------------------------------------------------------------  
            [ Ax, H ] = obj.FitModelContextObj.diagnosticPlots();   
            %--------------------------------------------------------------
            % Convert to natural unit scales
            %--------------------------------------------------------------
            Xdata = obj.X;
            Ydata = obj.Y;
            Xhi = obj.decodeX( H{1}(2).XData ).';
            Yhat = obj.decodeY( H{1}(2).YData ).';
            %--------------------------------------------------------------
            % Change the data on the fit diagnostic & relabel
            %--------------------------------------------------------------
            Hdl = H{ 1 };
            Hdl( 1 ).XData = Xdata;
            Hdl( 1 ).YData = Ydata;
            Hdl( 2 ).XData = Xhi;
            Hdl( 2 ).YData = Yhat;
            xlabel( Ax{ 1 }, obj.Xname );
            ylabel( Ax{ 1 }, obj.Yname )
            %--------------------------------------------------------------
            % Change the data on the weighted residual diagnostic & relabel
            %--------------------------------------------------------------
            H{ 2 }.XData = obj.decodeY( H{2}.XData );                     % Predictions are on the x-axis
            H{ 2 }.YData = obj.decodeY( H{2}.YData );
            Xlab = sprintf( '%s %s', "Predicted", obj.Yname );
            xlabel( Ax{ 2 }, Xlab );
            Ylab = sprintf( '%s %s', "Weighted Residual", obj.Yname );
            ylabel( Ax{ 2 }, Ylab );
            %--------------------------------------------------------------
            % Normal probability plot
            %--------------------------------------------------------------
            Hdl = H{ 3 };
            Hdl(1).XData = obj.decodeY( Hdl(1).XData );
            Hdl(2).XData = obj.decodeY( Hdl(2).XData );
            Hdl(3).XData = obj.decodeY( Hdl(3).XData );
            xlabel( Ax{ 3 }, Ylab );
            Ax{ 3 }.XLim = [min(Hdl(1).XData), max(Hdl(1).XData)];
            Ax{ 3 }.YLim = [min(Hdl(3).YData), max(Hdl(3).YData)];
            %--------------------------------------------------------------
            % Data vs. Predicted diagnostic
            %--------------------------------------------------------------
            Hdl = H{ 4 };
            Hdl(1).XData = obj.decodeY( Hdl(1).XData );                     % Predictions are on the x-axis
            Hdl(1).YData = obj.decodeY( Hdl(1).YData );                     % Predictions are on the y-axis
            Hdl(2).XData = obj.decodeY( Hdl(2).XData );                     % Predictions are on the x-axis
            Hdl(2).YData = obj.decodeY( Hdl(2).YData );                     % Predictions are on the y-axis
            Xlab = sprintf( '%s %s', "Predicted", obj.Yname );
            xlabel( Ax{ 4 }, Xlab );
            Ylab = sprintf( '%s %s', "Observed", obj.Yname );
            ylabel( Ax{ 4 }, Ylab );
            Ax{ 4 }.XLim = [min(Hdl(2).XData), max(Hdl(2).XData)];
            Ax{ 4 }.YLim = [min(Hdl(2).YData), max(Hdl(2).YData)];
        end
        
        
        function [ParameterVectors, Ave, StanDev] = bootStrapSamples( obj, Nboot )
            %--------------------------------------------------------------
            % Perform the bootstrap for the number of samples specified
            %
            % ParameterVectors = obj.bootStrapSamples( Nboot );
            %
            % Input Arguments:
            %
            % Nboot     --> Number of bootstrap samples {1000}
            %
            % Output Arguments:
            %
            % ParameterVectors  --> Parameter vectors for each bootstrap
            %                       sample ( obj.NumFitCoeff x N).
            % Ave               --> Average parameter vector from samples
            % StanDev           --> Standard deviation of samples
            %--------------------------------------------------------------
            if ( nargin < 2 )
                Nboot = 1000;
            end
            [ParameterVectors, Ave, StanDev] = obj.FitModelContextObj.bootStrapSamples( ...
                obj.NumCovPar, Nboot ); 
        end
    end % constructor and ordinary methods
    
    methods
        function Lam = get.Lamda( obj )
            % Return regularisation parameter
            Lam = obj.FitModelContextObj.Lamda;
        end
        
        function D = get.DoF( obj )
            % Return effective number of parameters
            D = obj.FitModelContextObj.DoF;
        end
        
        function M = get.Measure( obj )
            % Return information theoretic performance measure
            M = obj.FitModelContextObj.Measure;
        end
        
        function Xc = get.Xc( obj )
            % Return coded x-data (training)
            Xc = obj.codeX( obj.X );
        end
        
        function Yc = get.Yc( obj )
            % Return coded y-data (training)
            Yc = obj.codeY( obj.Y );
        end
        
        function D = get.Theta( obj )
            % Return fit parameter vector
            D = obj.FitModelContextObj.Delta;
        end
        
        function W = get.W( obj )
            % Return weights
            W = obj.FitModelContextObj.W;
        end
        
        function P = get.ParNames( obj )
            % Return the parameter names
            P = obj.FitModelContextObj.ParNames;
        end
        
        function M = get.ModelName( obj )
            % Return the fit model name
            M = obj.FitModelContextObj.ModelName;
        end
        
        function A = get.Algorithm( obj )
            % Return Lamda re-estimation algorithm
            A = obj.FitModelContextObj.Algorithm;
        end
        
        function N = get.N( obj )
            % Return number of data points
            N = obj.FitModelContextObj.N;
        end
        
        function D = get.Delta( obj )
            % Return covariance model parameters
            D = obj.CovModelContextObj.Delta;
        end
        
        function M = get.CovModel( obj )
            % Return covariance model name
            M = obj.CovModelContextObj.CovName;
        end
        
        function S = get.Sigma( obj )
            % Return variance scale standard error
            S = obj.CovModelContextObj.Sigma;
        end

        function S2 = get.Sigma2( obj )
            % Return variance scale parameter  
            S2 = obj.CovModelContextObj.Sigma2;
        end
        
        function C = get.NumCovPar( obj )
            % Number of covariance model parameters
            C = obj.CovModelContextObj.NumCoVParam;
        end
        
        function K = get.TotalNumPar( obj )
            % Total number of model parameters
            K = obj.DoF + obj.NumCovPar;
        end
    end % set/get methods
    
    methods ( Access = private )
        function X = decodeX( obj, Xc )
            %--------------------------------------------------------------
            % Map coded input data to natural units
            %
            % X = obj.decode( Xc );
            % 
            % Input Arguments:
            %
            % Xc    --> Coded input data
            %--------------------------------------------------------------
            [A, B, Ac, Bc] = obj.codeLimitsX();
            M = ( B - A )/( Bc - Ac );
            X = M*( Xc - Ac ) + A;
        end
        
        function Y = decodeY( obj, Yc )
            %--------------------------------------------------------------
            % Map coded observed data or predictions to natural units
            %
            % Y = obj.decode( Yc );
            % 
            % Input Arguments:
            %
            % Yc    --> Coded response data
            %--------------------------------------------------------------
            [A, B, Ac, Bc] = obj.codeLimitsY();
            M = ( B - A )/( Bc - Ac );
            Y = M*( Yc - Ac ) + A;
        end
        
        function C = codeX( obj, X )
            %--------------------------------------------------------------
            % Code the data to the prescribed range
            %
            % Map data from [DataLo, DataHi] --> [CodeLo, CodeHi]
            %
            % C = obj.code( X );
            %
            % Input Arguments:
            %
            % X     --> Input data in natural units
            %--------------------------------------------------------------
            [A, B, Ac, Bc] = obj.codeLimitsX();
            M = ( ( Bc - Ac )/( B - A ) );
            C = M*( X - A ) + Ac;
        end
        
        function [A, B, Ac, Bc] = codeLimitsX( obj )
            %--------------------------------------------------------------
            % Supply coding bound for input data
            %
            % [A, B, Ac, Bc] = obj.codeLimitsX();
            %--------------------------------------------------------------
            A = obj.XLB( 1 );
            B = obj.XUB( 1 );
            Ac = obj.XLB( 2 );
            Bc = obj.XUB( 2 );
        end
        
        function C = codeY( obj, Y )
            %--------------------------------------------------------------
            % Code the data to the prescribed range
            %
            % Map data from [DataLo, DataHi] --> [CodeLo, CodeHi]
            %
            % C = obj.code( Y );
            %
            % Input Arguments:
            %
            % Y     --> Observed data in natural units
            %--------------------------------------------------------------
            [A, B, Ac, Bc] = obj.codeLimitsY();
            M = ( ( Bc - Ac )/( B - A ) );
            C = M*( Y - A ) + Ac;
        end        
        
        function [A, B, Ac, Bc] = codeLimitsY( obj )
            %--------------------------------------------------------------
            % Supply coding bound for input data
            %
            % [A, B, Ac, Bc] = obj.codeLimitsY();
            %--------------------------------------------------------------
            A = obj.YLB( 1 );
            B = obj.YUB( 1 );
            Ac = obj.YLB( 2 );
            Bc = obj.YUB( 2 );
        end
    end % private methods
end


function mustBeFitModelObj( ModelObj )
    %----------------------------------------------------------------------
    % Validator function for ModelObj property
    %
    % mustBeFitModelObj( ModelObj )
    %----------------------------------------------------------------------
    if ~isempty( ModelObj ) && ~isa( ModelObj.Name,'RegFit.fitModelType' )
        error( 'Unrecognised fit model option' );
    end
end