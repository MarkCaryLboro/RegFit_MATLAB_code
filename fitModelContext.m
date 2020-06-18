classdef fitModelContext
    % RIGLS analysis class - provides user interface. 
    
    properties ( SetAccess = protected )
        fitModelObj                                                         % Fit model object
        W               double                                              % Data weight vector
        X               double                                              % Input data
        Y               double                                              % Observed data
    end
    
    properties ( SetAccess = protected, Dependent = true )
        Lamda                                                               % Regularisation parameter
        DoF                                                                 % Model degree of freedom including covariance parameters
        Algorithm                                                           % Re-estimation algorithm name
        ModelName                                                           % Fit model type
        N                                                                   % Number of data points
        ParNames                                                            % Parameter names
        Delta                                                               % Fit parameter vector
        Measure                                                             % Information theoretic performance measure
    end
    
    methods
        function obj = fitModelContext( fitModelObj, X, Y, W )
            %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.fitModelContext( fitModelObj, X, Y, W );
            %
            % Input Arguments:
            %
            
            % fitModelObj   --> A RegFit.fitModel object
            % X             --> Input data
            % Y             --> Observed data
            % W             --> Data weights {1}
            %--------------------------------------------------------------
            if ( nargin < 4 )
                W = ones( size( X ) );                                      % Apply default
            end
            obj.W = W;
            obj.X = X;
            obj.Y = Y;
            obj.fitModelObj = fitModelObj;
        end
        
        function obj = nonLinRegFit( obj, NumCovPar, Options )
            %--------------------------------------------------------------
            % Fit the data
            %
            % obj = obj.nonLinRegFit( NumCovPar, Options );
            %
            % Input Arguments:
            %
            % NumCovPar --> Number of covariance parameters
            % Options   --> Optimisation configuration object. Create with
            %               Options = optimoptions( 'fmincon' );
            %--------------------------------------------------------------
            if ( nargin < 3)
                Options = optimoptions( 'fmincon' );
                Options.Display = 'None';
            end
            Options.SpecifyObjectiveGradient = true;
            obj.fitModelObj = obj.fitModelObj.mleRegTemplate( obj.X, obj.Y, obj.W, NumCovPar, Options );
        end
        
        function [ Ax,  H ] = diagnosticPlots( obj )
            %--------------------------------------------------------------
            % Model diagnostic plots
            %
            % obj.diagnosticPlots();
            %--------------------------------------------------------------
            [ Ax, H ] = obj.fitModelObj.diagnosticPlots( obj.X, obj.Y, obj.W );
        end
        
        function J = jacobean( obj, X )
            %-------------------------------------------------------------- 
            % Return jacobean matrix
            %
            % J = obj.jacobean( X );
            %
            % Input Arguments:
            %
            % X     --> Input data {obj.X}
            %--------------------------------------------------------------
            if ( nargin < 2 )
                X = obj.X;
            end
            J = obj.fitModelObj.jacobean( X );
        end
        
        function SE = stdErrors( obj )
            %--------------------------------------------------------------
            % Calculate the standard errors for the parameter estimates
            %
            % SE = obj.stdErrors( );
            %
            %-------------------------------------------------------------- 
            SE = obj.fitModelObj.stdErrors( obj.X, obj.W );
        end
        
        function P = predictions( obj, X )
            %--------------------------------------------------------------
            % Model predictions
            %
            % P = obj.predictions( X );
            %
            % Input Arguments:
            %
            % X     --> Input data {obj.X}
            %--------------------------------------------------------------
            if ( nargin < 2 )
                X = obj.X;
            end
            P = obj.fitModelObj.predictions( X );
        end
        
        function obj = setWeights( obj, W )
            %--------------------------------------------------------------
            % Set the weight vector
            %
            % obj = obj.setWeights( W );
            %
            % Input Arguments:
            %
            % W     --> Weight vector
            %--------------------------------------------------------------
            if ( nargin < 2 ) || ( numel( W ) ~= obj.N )
                error('Weight Vector not Assigned');
            else
                obj.W = W;
            end
        end
        
        function [ParameterVectors, Ave, StanDev] = bootStrapSamples( obj, NumCovPar, Nboot )
            %--------------------------------------------------------------
            % Perform the bootstrap for the number of samples specified
            %
            % ParameterVectors = obj.bootStrapSamples( NumCovPar, Nboot );
            %
            % Input Arguments:
            %
            % NumCovPar --> Number of covariance parameters
            % Nboot     --> Number of bootstrap samples {250}
            %
            % Output Arguments:
            %
            % ParameterVectors  --> Parameter vectors for each bootstrap
            %                       sample ( obj.NumFitCoeff x N).
            % Ave               --> Average parameter vector from samples
            % StanDev           --> Standard deviation of samples
            %--------------------------------------------------------------
            if ( nargin < 3 )
                Nboot = 250;
            end
            [ParameterVectors, Ave, StanDev] = obj.fitModelObj.bootStrapSamples( obj.X,...
                obj.Y, obj.W, NumCovPar, Nboot ); 
        end
    end % constructor and ordinary methods
    
    methods
        function N = get.N( obj )
            % Return number of data points
            N = numel( obj.X );
        end
        
        function M = get.ModelName( obj )
            % Return fit model type
            M = obj.fitModelObj.ModelName;
        end
        
        function Alg = get.Algorithm( obj )
            % Return lamda re-estimation algorithm
            Alg = obj.fitModelObj.ReEstObj.Algorithm;
        end
        
        function Lam = get.Lamda( obj )
            % Return regularisation parameter
            Lam = obj.fitModelObj.Lamda;
        end
        
        function D = get.DoF( obj )
            % Return model parameter count
            D = obj.fitModelObj.ReEstObj.DoF;
        end
        
        function M = get.Measure( obj )
            % Return information theoretic performance measure
            M = obj.fitModelObj.ReEstObj.Measure;
        end
        
        function P = get.ParNames( obj )
            % Return parameter names
            P = obj.fitModelObj.ParNames;
        end
        
        function D = get.Delta( obj )
            % Return fit parameter vector
            D = obj.fitModelObj.Theta;
        end
    end % set/get methods
    
    methods ( Hidden = true )
    end % hidden methods
end


