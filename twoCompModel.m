classdef twoCompModel < RegFit.covModel
    % Two-components of variance model
    
    properties 
        Delta   double   = [1, 0.5]                                         % Power law exponent
    end
    
    properties ( Constant = true )
        CovName    RegFit.covModelType = "TwoComponents"                    % Covariance model name
        LB         double = [0, 0]                                          % Lower bound for Delta
        UB         double = [5, 5]                                          % Upper bound for Delta
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
            W = obj.Delta( 1 ) + Yhat.^( 2*obj.Delta( 2 ));
        end
    end % end ordinary and constructor methods
    
    methods 
        function obj = set.Delta( obj, Value )
            if any( isnan( Value ) )
                obj.Delta = nan( obj.NumCoVParam, 1 );
            else
                ValueFlag = isreal( Value ) & isnumeric( Value ) & ( Value >= obj.LB ) & ( Value <= obj.UB );
                if all( ValueFlag )
                    obj.Delta = double( Value );
                else
                    error('[Class]: RegFit.powerModel, [Method]:set.Delta,: [Msg]: "Value not assigned"');
                end
            end
        end
    end % set/get methods
end
