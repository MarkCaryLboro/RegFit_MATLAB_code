classdef covModelType < int8
    % Enumeration class for supported covariance model types. You need to edit this if you add additional covariance models.
    
    enumeration
        OLS             (0)
        Power           (1)
        Exponential     (2)
        TwoComponents   (3)
    end
end