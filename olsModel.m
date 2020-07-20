classdef olsModel < RegFit.covModel
    % Ordinary least squares cost covariance model
    
    properties 
        Delta   double   = []                                               % Power law exponent
    end    
    
    properties ( Constant = true )
        CovName    RegFit.covModelType = "OLS"                              % Covariance model name
        LB         double = []                                              % Lower bound for Delta
        UB         double = []                                              % Upper bound for Delta
    end    
    
    methods
        function W = calcWeights( obj, Yhat ) %#ok<INUSL>
            %--------------------------------------------------------------
            % Calculate the weights
            %
            % W = obj.calcWeights( Yhat );
            %
            % Input Arguments:
            %
            % Yhat  --> Fit model predictions
            %--------------------------------------------------------------
            W = ones( size( Yhat ) );
        end
    end % end ordinary and constructor methods    
    
    methods 
        function obj = set.Delta( obj, Value ) %#ok<INUSD>
            % Always return an empty value
            obj.Delta = [];
        end
    end % set/get methods    
end