classdef covModelContext
    % covariance modelling context class - provides user interface
    properties ( SetAccess = immutable )
        Y           double                                                  % Observed data
    end
    
    properties ( SetAccess = protected )
        ModelObj    {mustBeCovModel( ModelObj )}                            % covModel object
        Yhat        double                                                  % Model predictions
    end
    
    properties ( SetAccess = protected, Dependent = true )
        NumCoVParam                                                         % Number of covariance parameters
        Sigma2                                                              % Variance scale parameter
        Sigma                                                               % Standard error scale parameter
        Delta                                                               % Covariance model parameters
        N                                                                   % Number of data points
        CovName                                                             % Covariance model name
    end
    
    methods
        function obj = covModelContext( ModelObj, Y, Yhat )
            %--------------------------------------------------------------
            %
            % Class constructor
            %
            % obj = RegFit.covModelContext( ModelObj, Y );
            % obj = RegFit.covModelContext( ModelObj, Y, Yhat );
            %
            % Input Arguments:
            %
            % ModelObj  --> RegFit.covModel object (covariance parameter
            %               estimation).
            % Y         --> Observed data
            % Yhat      --> Fit model estimates
            %--------------------------------------------------------------
            obj.ModelObj = ModelObj;
            obj.Y = Y;
            if ( nargin > 2 )
                obj.Yhat = Yhat;
            end
        end
        
        function obj = profileLikelihood( obj )
            %--------------------------------------------------------------
            % Identify the covariance model parameters
            %
            % obj = obj.profileLikelihood();
            %--------------------------------------------------------------
            obj.ModelObj = obj.ModelObj.mleTemplate( obj.Y, obj.Yhat );
        end
        
        function plotProfileLikelihood( obj, NumPts )
            %--------------------------------------------------------------
            % Plot the profile likelihood between the limits specified
            %
            % obj.plotProfileLikelihood( NumPts )
            %
            % Input Arguments:
            %
            % NumPts    --> Number of samples for each parameter {101}
            %--------------------------------------------------------------
            if nargin<2
                NumPts = 101;                                               % Apply default
            end
            obj.ModelObj.plotProfileLikelihood( obj.Y, obj.Yhat, NumPts );
        end
        
        function W = calcWeights( obj )
            %--------------------------------------------------------------
            % Calculate the variance weights
            %
            % W = obj.calcWeights();
            %--------------------------------------------------------------
            W = obj.ModelObj.calcWeights( obj.Yhat );
        end
        
        function obj = setPredictions( obj, Yhat )
            %--------------------------------------------------------------
            % Set the prediction vector
            %
            % obj = obj.setPredictions( Yhat );
            %
            % Input Arguments:
            %
            % Yhat  --> Predicted response vector
            %--------------------------------------------------------------
            if isempty(obj.Y) || ( numel( Yhat ) == obj.N )
                obj.Yhat = reshape( Yhat, (size ( obj.Y ) ) );
            else
                ErrMesg = '[Class]: "RegFit.covModelContext", [Method]: "setPredictions", [Message]: "Incorrect dimension for prediction vector, value not set!"';
                error( ErrMesg );
            end
        end
    end % constructor and ordinary methods
    
    methods
        function N = get.NumCoVParam( obj )
            % Number of covariance parameters
            N = obj.ModelObj.NumCoVParam + 1;
        end
        
        function D = get.Delta( obj )
            % Return covariance parameter vector
            D = obj.ModelObj.Delta;
        end
        
        function S2 = get.Sigma2( obj )
            % Variance scale parameter
            S2 = obj.ModelObj.Sigma2;
        end
        
        function S = get.Sigma( obj )
            % Standard error scale parameter
            S = obj.ModelObj.Sigma;
        end
        
        function N = get.N( obj )
            % Return number of data points
            N = numel( obj.Y );
        end
        
        function M = get.CovName( obj )
            % Return covariance model name
            M = obj.ModelObj.CovName;
        end
    end % get/set methods
end

function mustBeCovModel( ModelObj )
    %----------------------------------------------------------------------
    % Validator function for ModelObj property
    %
    % mustBeCovModel( ModelObj )
    %----------------------------------------------------------------------
    Names = ["Power", "OLS", "Exponential", "TwoComponents"];
    if ~isempty( ModelObj ) && ~contains( string( ModelObj.CovName ), Names )
        error('Unrecognised covariance model option');
    end
end