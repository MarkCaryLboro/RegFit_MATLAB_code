classdef aicReEst < RegFit.reEstLamda
    % Re-estimate lamda based on small sample AIC
    properties ( SetAccess = protected )
        Measure                                                             % BIC value
    end
    
    properties ( Constant = true )
        Name = "aicReEst"
    end
    
    methods
        function obj = getMeasure( obj, Lam, Res, W, J, NumCovPar )
            %--------------------------------------------------------------
            % Calculate the current Small Sample AIC
            %
            % obj = obj.getMeasure( Lam, Res, W, J, NumCovPar );
            %
            % Input Arguments:
            %
            % Lam       --> Starting value for lamda re-estimation
            % Res       --> Residual vector
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % NumCovPar --> Number of covariance model parameters
            %--------------------------------------------------------------
            [Lp, S] = obj.calcProfileLikelihood( Lam, Res, W, J );
            D = trace( S ) + NumCovPar;
            N = numel( Res );
            obj.Measure = 2*Lp + 2*D*( D + 1 )/( N - D - 1);
        end
    end % ordinary methods
    
    methods ( Access = protected )
        function Lam = calculateLamda( obj, Lam, Res, W, J, NumCovPar, MaxIter )
            %--------------------------------------------------------------
            % Calculate the next value for the hyper-parameter
            %
            % Lam = obj.calculateLamda( Lam, Res, W, J, NumCovPar, MaxIter );
            %
            % Input Arguments:
            %
            % Lam       --> Starting value for lamda re-estimation
            % Res       --> Residual vector
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % NumCovPar --> Number of covariance model parameters
            % MaxIter   --> Maximum number of iteration {10}. MaxIter is
            %               clipped to 1 as a minimum.
            %--------------------------------------------------------------
            if (nargin < 7) || isempty( MaxIter )
                MaxIter = 10;                                               % Apply default
            end
            %--------------------------------------------------------------
            % Re-estimate lamda
            %--------------------------------------------------------------
            stopflg = false;
            Iter = 0;                                                       % Iteration counter
            while ~stopflg
                Iter = Iter + 1;                                            % Current iteration
                LastLam = Lam;                                              % Last value of regularisation parameter
                Lam = obj.calcNewLam( W, J, Res, Lam, NumCovPar );
                %----------------------------------------------------------
                % Clip Lamda to effectively zero
                %----------------------------------------------------------
                Lam = max([ sqrt(eps), Lam]);
                %----------------------------------------------------------
                % Apply convergence test
                %----------------------------------------------------------
                ConvCriteria = 100*abs( ( Lam - LastLam )/LastLam );        % Convergence criteria
                stopflg = ( Iter>=MaxIter ) | ( ConvCriteria<0.01 );        % Stopping criterion
            end
        end
        
        function NewLam = calcNewLam( obj, W, J, Res, Lam, NumCovPar )
            %--------------------------------------------------------------
            % Calculate new value of hyper-parameter
            %
            % NewLam = obj.calcNewLam( W, J, Res, Lam, NumCovPar );
            %
            % Input Arguments:
            %
            % Lam       --> Current hyper-parameter estimate
            % Res       --> Residual vector
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % NumCovPar --> Number of covariance model parameters
            %--------------------------------------------------------------
            N = numel( Res );                                               % Number of data points
            C = sqrt( W );                                                  % Cholesky factor for W;
            Q = Res./C;                                                     % Weighted residual vector
            [S, Z, ~, IA] = obj.calcSmatrix( Lam, W, J );                   % Return relevant matrices
            Sigma2 = Q.'*( eye( N ) - S )^2*Q/N;                            % Variance scale parameter
            G = trace( S ) + NumCovPar;                                     % Total number of parameters
            M = 2*( 2*G + 1 )./(N - G - 1) - ...
                2*G*( G + 1 )./(N - G - 1)^2;
            DLdLam = Q.'*Z*(IA^3)*Z.'*Q;                                    % Derivative of the likelihood with respect to Lamda
            DdoFdLam = trace( IA - Lam*IA^2 );                              % Derivative of DoF with respect to Lamda
            NewLam = Sigma2*M*DdoFdLam./2./N./DLdLam;                       % Updated hyper-parameter estimate
        end        
    end % protected methods
    
    methods ( Access = private )
    end % private methods
end