classdef mrm < RegFit.fitModel
    % Relaxation model due to Meng, J., Stroe, D.I., Ricco, M., Luo, G.,...
    % Swierczynski, M. and Teodorescu, R., 2018. A novel multiple ...
    % correction approach for fast open circuit voltage prediction of ...
    % lithium-ion battery. IEEE Transactions on Energy Conversion,...
    % 34(2), pp.1115-1123.

    properties
        Theta       double                                                  % Model parameter vector
        LB          double                                                  % Lower bound constraint for parameters
        UB          double                                                  % Upper bound constraint for parameters
    end
    
    
end % mrm class