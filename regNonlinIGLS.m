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
        FitModelObj         { mustBeFitModelObj( FitModelObj ) }            % Regularised fit model object
        CovModelObj         { mustBeCovModelObj( CovModelObj ) }            % Covariance model context object
        W                   double                                          % Data weights for IGLS analysis
    end
    
    properties ( Access = private )
        N_                   double                                         % Number of points used for fitting
        X_                   double                                         % Regressor vector
        Y_                   double                                         % Observed data vector
    end % private properties
    
    properties ( SetAccess = protected, Dependent = true )
        Lamda               double                                          % Regularisation coefficient
        DoF                 double                                          % Effective number of parameters
        Algorithm                                                           % Re-estimation algorithm name
        ModelName                                                           % Fit model type
        N                                                                   % Number of data points
        ParNames            string                                          % Parameter names
        Theta               double                                          % Fit parameter vector
        Xc                  double                                          % Coded regressor data
        Yc                  double                                          % Coded observed data
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
            obj.FitModelObj = fitModelObj;
            obj.CovModelObj = covModelObj;
        end
        
        function C = codeData( obj, Data, Type )
            %--------------------------------------------------------------
            % Code either X- or Y-data.
            %
            % C = obj.codeData( Data, Type );
            %
            % Input Arguments:
            %
            % Data  --> (double) Vector of data points in natural units
            % Type  --> (string) Either {"X"} for X-data or "Y" for Y-Data
            %--------------------------------------------------------------
            arguments
                obj     (1,1)
                Data       (:,1)   double        { mustBeNonempty( Data ) }
                Type    (1,1)   string  = "X"
            end
            if strcmpi( Type, "x" ) 
                Type = "X";
            else
                Type = "Y";
            end
            switch Type
                case "X"
                    C = obj.codeX( Data );
                otherwise
                    C = obj.codeY( Data );
            end
        end % codeData
        
        function Data = decodeData( obj, C, Type )
            %--------------------------------------------------------------
            % Decode either coded X- or Y-data.
            %
            % Data = obj.decodeData( C, Type );
            %
            % Input Arguments:
            %
            % C     --> (double) Vector of data points in coded units
            % Type  --> (string) Either {"X"} for X-data or "Y" for Y-Data
            %--------------------------------------------------------------
            arguments
                obj     (1,1)
                C       (:,1)   double        { mustBeNonempty( C ) }
                Type    (1,1)   string  = "X"
            end
            if strcmpi( Type, "x" ) 
                Type = "X";
            else
                Type = "Y";
            end
            switch Type
                case "X"
                    Data = obj.decodeX( C );
                otherwise
                    Data = obj.decodeY( C );
            end            
        end % decodeData
        
        
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
            obj.W = ones( obj.N, 1);         
            %--------------------------------------------------------------
            % Remove any aberrant data
            %--------------------------------------------------------------
            if strcmpi( "PRM", obj.ModelName )
                %----------------------------------------------------------
                % TO DO - Need to alter architecture to allow parseInputs
                % method to be overloaded
                %----------------------------------------------------------
                [ obj.X_, obj.Y_, obj.W ] = obj.FitModelObj.parseInputs( obj.X,...
                    obj.Y, obj.W, obj.FitModelObj.Tct );
            else
                [ obj.X_, obj.Y_, obj.W ] = obj.FitModelObj.parseInputs( obj.X,...
                                                obj.Y, obj.W );
            end
            %--------------------------------------------------------------
            % ROLS fit
            %--------------------------------------------------------------
            obj.FitModelObj = obj.FitModelObj.mleRegTemplate( obj.Xc,...
                obj.Yc, obj.W, obj.NumCovPar, Options );
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
                fprintf( '\nIGLS Iteration #%d', Iter );
                [ ~, Yhat ] = obj.predictions( obj.X_ );
                obj.CovModelObj = obj.CovModelObj.mleTemplate( obj.Yc, Yhat );
                obj.W = obj.CovModelObj.calcWeights( Yhat );
                obj.FitModelObj = obj.FitModelObj.mleRegTemplate( obj.Xc,...
                    obj.Yc, obj.W, obj.NumCovPar, Options );    
                fprintf( ' - lambda = %6.4e', obj.Lamda );
                ConvFlg = 100*( norm(obj.Theta - ThetaLast )/norm( obj.Theta ) ) <= 0.0001;
                Stopflg = ConvFlg | ( Iter >= MaxIter );
            end
            fprintf('\n\n');
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
                X = obj.X_;                                                 % Apply default
            end 
            X = X(:);
            C = obj.codeX( X );
            J = obj.FitModelObj.jacobean( C );
        end % jacobean
        
        function [ Res,  ResC ] = calcResiduals( obj )
            %--------------------------------------------------------------
            % Return training residuals
            %
            % [ Res, ResC ] = obj.calcResiduals();
            %
            % Output Arguments:
            %
            % Res   --> Residuals in natural units
            % ResC  --> Residuals in coded units
            %--------------------------------------------------------------
            ResC = obj.FitModelObj.calcResiduals( obj.Xc, obj.Yc );
            [A, B, Ac, Bc] = obj.codeLimitsY();
            M = ( B - A )/( Bc - Ac );
            Res = M * ResC ;
        end % calcResiduals
        
        function [ Yhat, YhatC, X ] = predictions( obj, X )
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
                X = obj.X_;
            end
            switch strcmpi( obj.ModelName, "PRM" )
                case true
                    [ Yhat, YhatC, X ] = obj.makePRMpredictions( X );
                otherwise
                    [ Yhat, YhatC ] = obj.makePredictions( X );
            end
        end % predictions
        
        function SE = stdErrors( obj )
            %--------------------------------------------------------------
            % Return standard errors for the fit parameters
            %
            % SE = obj.stdErrors();
            %
            %--------------------------------------------------------------
            SE = obj.FitModelObj.stdErrors( obj.Xc, obj.W );
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
            DF = obj.N_ - obj.TotalNumPar;
            SE = obj.Sigma*obj.stdErrors();
            T = abs(tinv( 0.5*P, DF ));                                     % T-statistic
            LCI = obj.Theta - SE*T;                                         % Lower C.I.
            UCI = obj.Theta + SE*T;                                         % Upper C.I.
        end
        
        function [Ax, H] = diagnosticPlots( obj )
            %--------------------------------------------------------------
            % Model diagnostic plots
            %
            % obj.diagnosticPlots();
            %
            % Output Arguments:
            %
            % Ax            --> Axes handles
            % H             --> Line handles
            %--------------------------------------------------------------
            figure;
            Ax{ 1 } = subplot( 2, 2, 1 );
            H{ 1 } = obj.fitsPlot( Ax{ 1 } );
            Ax{ 2 } = subplot( 2, 2, 2 );
            H{ 2 } = obj.weightedResPlot( Ax{ 2 } );
            Ax{ 3 } = subplot( 2, 2, 3 );
            H{ 3 } = obj.normalPlot( Ax{ 3 } );
            Ax{ 4 } = subplot( 2, 2, 4 );
            H{ 4 } = obj.dataVsPredPlot( Ax{ 4 } );
            for Q = 1:numel( Ax )
                %----------------------------------------------------------
                % Make the grid more visible and 
            end
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
            [ParameterVectors, Ave, StanDev] = obj.FitModelObj.bootStrapSamples( ...
                obj.Xc, obj.Yc, obj.W, obj.NumCovPar, Nboot ); 
        end % bootStrapSamples
    end % constructor and ordinary methods
    
    methods
        function Lam = get.Lamda( obj )
            % Return regularisation parameter
            Lam = obj.FitModelObj.Lamda;
        end
        
        function D = get.DoF( obj )
            % Return model degrees of freedom
            D = obj.FitModelObj.ReEstObj.DoF;
        end
        
        function M = get.Measure( obj )
            % Return information theoretic performance measure
            M = obj.FitModelObj.Measure;
        end
        
        function Xc = get.Xc( obj )
            % Return coded x-data (training)
            if strcmpi( "PRM", obj.ModelName )
                Xc( :, 1 ) = obj.codeX( obj.X_( :,1 ) );
                Xc( :, 2 ) = obj.codeY( obj.X_( :,2 ) );
            else
                Xc = obj.codeX( obj.X_ );
            end
        end
        
        function Yc = get.Yc( obj )
            % Return coded y-data (training)
            Yc = obj.codeY( obj.Y_ );
        end
        
        function T = get.Theta( obj )
            % Return fit parameter vector
            T = obj.FitModelObj.Theta;
        end
        
        function P = get.ParNames( obj )
            % Return the parameter names
            P = obj.FitModelObj.ParNames;
        end
        
        function M = get.ModelName( obj )
            % Return the fit model name
            M = obj.FitModelObj.ModelName;
        end
        
        function A = get.Algorithm( obj )
            % Return Lamda re-estimation algorithm
            A = obj.FitModelObj.Algorithm;
        end
        
        function N = get.N( obj )
            % Return number of data points
            N = numel( obj.X );
        end
        
        function N = get.N_( obj )
            % Return number of data points used in fitting
            N = numel( obj.X_( :,1 ) );
        end
        
        function D = get.Delta( obj )
            % Return covariance model parameters
            D = obj.CovModelObj.Delta;
        end
        
        function M = get.CovModel( obj )
            % Return covariance model name
            M = obj.CovModelObj.CovName;
        end
        
        function S = get.Sigma( obj )
            % Return variance scale standard error
            S = obj.CovModelObj.Sigma;
        end

        function S2 = get.Sigma2( obj )
            % Return variance scale parameter  
            S2 = obj.CovModelObj.Sigma2;
        end
        
        function C = get.NumCovPar( obj )
            % Number of covariance model parameters
            C = obj.CovModelObj.NumCoVParam;
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
        
        function H = dataVsPredPlot( obj, Ax )
            %--------------------------------------------------------------
            % Data versus predictions plot diagnostic
            %
            % H = obj.dataVsPredPlot( Ax );
            %
            % Input Arguments:
            %
            % Ax        --> Axes handle
            %
            % Output Arguments:
            %
            % H         --> Handle to line objects 
            %--------------------------------------------------------------
            Xd = obj.X;
            if strcmpi( obj.ModelName, "PRM" )
                %----------------------------------------------------------
                % Augment X to ensure consistency of dimensions of
                % predictions and data
                %----------------------------------------------------------
                Xd = [ Xd( 1 ) - 1/120; Xd ];
            end
            Yhat = obj.predictions( Xd );
            H = plot( Yhat, obj.Y, 'bo' );
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
            lab = sprintf('Predicted %s', obj.Yname);
            xlabel( lab );
            lab = sprintf('Observed %s', obj.Yname);
            ylabel( lab);
            title('Data Versus Predicted');
        end
        
        function H = normalPlot( obj, Ax ) 
            %--------------------------------------------------------------
            % Normal probability plot for weighted residuals
            %
            % H = obj.normalPlot( Ax );
            %
            % Input Arguments:
            %
            % Ax        --> Axes handle
            %
            % Output Arguments:
            %
            % H         --> Handle to line objects 
            %--------------------------------------------------------------
            Res = obj.calcResiduals();
            Res = Res./sqrt( obj.W );
            H = normplot( Ax, Res );
            lab = sprintf('Weighted Residual %s', obj.Yname);
            xlabel( lab );
        end
        
        function H = weightedResPlot( obj, Ax ) 
            %--------------------------------------------------------------
            % Weighted residual versus predicted plot
            %
            % H = obj.weightedResPlot( Ax );
            %
            % Input Arguments:
            %
            % Ax        --> Axes handle
            %
            % Output Arguments:
            %
            % H         --> Handle to line objects 
            %--------------------------------------------------------------
            Res = obj.calcResiduals();
            Yhat = obj.predictions();
            Res = Res./sqrt( obj.W );
            H = plot( Ax, Yhat, Res, 'bo' );
            H.MarkerFaceColor = 'blue';
            grid on;
            lab = sprintf('Predicted %s', obj.Yname);
            xlabel( lab );
            lab = sprintf('Weighted Residual %s', obj.Yname);
            ylabel( lab );
            title('Weighted Residual Vs. Prediction');
        end
        
        function H = fitsPlot( obj, Ax, NumPts)
            %--------------------------------------------------------------
            % Model fits to data
            %
            % H = obj.fitsPlot( Ax, NumPts );
            %
            % Input Arguments:
            %
            % Ax        --> Axes handle
            % NumPts    --> Number of hi res points for fitted line {101}
            %
            % Output Arguments:
            %
            % H         --> Handle to line objects 
            %--------------------------------------------------------------
            if ( nargin < 5 ) || ( NumPts < obj.N )
                NumPts = 101;
            end
            Xhi = linspace( min( obj.X ), max( obj.X ), NumPts ).';
            [Yhi, ~, Xhi] = obj.predictions( Xhi );
            H = plot( Ax, obj.X, obj.Y, 'bo', Xhi, Yhi, 'r-' );
            H(1).MarkerFaceColor = 'blue';
            H(2).LineWidth = 2.0;
            grid on
            xlabel( obj.Xname );
            ylabel( obj.Yname );
            title('Model Fits')
        end % fitsPlot
        
        function [ Yhat, YhatC ] = makePredictions( obj, X )
            %--------------------------------------------------------------
            % Prediction calculations for the non-PRM case
            %            
            % [ Yhat, YhatC ] = obj.makePredictions( X );
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
            C = obj.codeX( X );
            YhatC = obj.FitModelObj.predictions( C );
            Yhat = obj.decodeY( YhatC );
        end % makePredictions
        
        function [ Yhat, YhatC, X ] = makePRMpredictions( obj, X )
            %--------------------------------------------------------------
            % Prediction calculations for the PRM case
            %            
            % [ Yhat, YhatC ] = obj.makePRMpredictions( X );
            %
            % Input Arguments:
            %
            % X     --> Regressor vector in natural units {obj.X}
            %
            % Output Arguments:
            %
            % Yhat  --> Predictions in natural units
            % YhatC --> Predictions in coded units
            % X     --> Adjusted length due to prediction process
            %-------------------------------------------------------------- 
            if ( numel( X ) == 1 )
                %----------------------------------------------------------
                % Add an extra data point 30 seconds before if only one piece
                % of data
                %----------------------------------------------------------
                X = [ X - 1/120; X ];
            end
            %--------------------------------------------------------------
            % Interpolate the corresponding Y-data
            %--------------------------------------------------------------
            Ycode = interp1( obj.X, obj.Y, X, 'linear' );
            Xcode = obj.codeX( X );
            Ycode = obj.codeY( Ycode );
            Xcode = [ Xcode, Ycode ];
            YhatC = obj.FitModelObj.predictions( Xcode );
            Yhat = obj.decodeY( YhatC );
            X = X( 2:end );
        end % makePRMpredictions
    end % private methods
end

function mustBeCovModelObj( ModelObj )
    %----------------------------------------------------------------------
    % Validator function for ModelObj property
    %
    % mustBeCovModelObj( ModelObj )
    %----------------------------------------------------------------------
    if ~isempty( ModelObj ) && ~isa( ModelObj.CovName,'RegFit.covModelType' )
        error( 'Unrecognised covariance model option' );
    end
end

function mustBeFitModelObj( ModelObj )
    %----------------------------------------------------------------------
    % Validator function for ModelObj property
    %
    % mustBeFitModelObj( ModelObj )
    %----------------------------------------------------------------------
    if ~isempty( ModelObj ) && ~isa( ModelObj.ModelName,'RegFit.fitModelType' )
        error( 'Unrecognised fit model option' );
    end
end