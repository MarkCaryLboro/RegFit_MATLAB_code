classdef bicReEst < RegFit.reEstLamda
    % Re-estimate lamda based on BIC
    
    properties ( Constant = true )
        Name        RegFit.reEstType = "BIC"
    end
    
    properties ( SetAccess = protected )
        Measure                                                             % BIC value
    end
    
    methods
        function obj = getMeasure( obj, Lam, Res, W, J, NumCovPar )
            %--------------------------------------------------------------
            % Calculate the current Bayesian Information Criterion
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
            N = numel( Res );
            obj.Measure = 2*Lp + log( N )*( trace( S ) + NumCovPar );
        end

        function DHDL = firstDerivative( obj, W, J, Res, Lam, ~ )
            %--------------------------------------------------------------
            % Calculate dh(Lam)/dLam using 5-point stencil formula
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
            C = sqrt( W );                                                  % Cholesky factor for W;
            R = Res./C;                                                     % Weighted residual vector
            N = numel( Res );                                               % Number of data points
            [S, Z, ~, IA] = obj.calcSmatrix( Lam, W, J );                   % Return relevant matrices
            Sigma2 = R.'*( eye( N ) - S )^2*R / N;                          % Variance scale parameter
            U = trace( IA - Lam * IA^2 );
            DUDL = 2 * trace( Lam * IA^3 - IA^2 );
            V = R.' * Z * IA^3 * Z.' * R;
            DVDL = -3 * R.' * Z * IA^4 * Z.' * R;
            Mul = Sigma2 * log( N ) / N / 2;
            DHDL = Mul * ( V * DUDL - U * DVDL )/ V^2;
        end % firstDerivative
    end % ordinary methods
    
    methods ( Access = protected )
        function Lam = calculateLamda( obj, Lam, Res, W, J, NumCovPar, MaxIter ) 
            %--------------------------------------------------------------
            % re-estimate Lamda based on BIC
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
                % Clip Lamda to zero
                %----------------------------------------------------------
                Lam = max([ 0, Lam ]);
                %----------------------------------------------------------
                % Apply convergence test
                %----------------------------------------------------------
                ConvCriteria = 100*abs( ( Lam - LastLam )/LastLam );        % Convergence criteria
                stopflg = ( Iter>=MaxIter ) | ( ConvCriteria<0.0001 );      % Stopping criterion
            end
        end
        
        function NewLam = calcNewLam( obj, W, J, Res, Lam, NumCovPar ) %#ok<INUSD>
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
            DLdLam = Q.'*Z*(IA^3)*Z.'*Q;                                    % Derivative of the likelihood with respect to Lamda
            DdoFdLam = trace( IA - Lam*IA^2 );                              % Derivative of DoF with respect to Lamda
            NewLam = Sigma2*log(N)*DdoFdLam/DLdLam/2/N;                     % Updated hyper-parameter estimate
        end        
    end % protected methods
end