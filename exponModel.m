classdef exponModel < RegFit.covModel
    % Exponential Covariance Model for Heteroscedastic Data
    
    
    properties 
        Delta   double   = 1.0                                              % Power law exponent
    end
    
    properties ( Constant = true )
        CovName    RegFit.covModelType = "Exponential"                      % Covariance model name
        LB         double = 0                                               % Lower bound for Delta
        UB         double = 5                                               % Upper bound for Delta
    end
    
    methods
        function W = calcWeights( obj, Yhat )
            %--------------------------------------------------------------
            % Calculate the weights
            %
            % W = obj.calcWeights( Yhat );
            %
            % Input Arguments:
            %
            % Yhat  --> Fit model predictions
            %--------------------------------------------------------------
            W = exp( 2*obj.Delta*Yhat );
        end
    end % end ordinary and constructor methods
    
    methods 
        function obj = set.Delta( obj, Value )
            if isnan(Value)
                obj.Delta = nan;
            elseif isreal( Value ) && isnumeric( Value ) && ( Value >= obj.LB ) && ( Value <= obj.UB )
                obj.Delta = double( Value );
            else
                error('[Class]: RegFit.powerModel, [Method]:set.Delta,: [Msg]: "Value not assigned"');
            end
        end
    end % set/get methods    
    
end