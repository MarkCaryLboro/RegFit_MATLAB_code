classdef reEstLamda
    % Re-estimation of regularisation parameter for ridge regression
    properties ( SetAccess = protected, Abstract = true )
        Measure                                                             % Information theoretic measure
    end
    
    properties ( SetAccess = protected )
        Lamda   double {mustBeGreaterThanOrEqual( Lamda, 0 ),...            % Regularisation parameter
                        mustBeFinite( Lamda ), mustBeReal( Lamda )} = 0.001
        DoF     double {mustBeGreaterThan( DoF, 0 ),...                     % Effective number of parameters
                        mustBeFinite( DoF ), mustBeReal( DoF )}
    end
    
    methods ( Abstract = true )
        obj = getMeasure( obj, Lam, Res, W, J, NumCovPar )
    end % Abstract method signatures
    
    methods ( Access = protected, Abstract = true )
        Lam = calculateLamda( obj, Q, Z, MaxIter )
        NewLam = calcNewLam( obj, W, J, Res, Lam, NumCovPar )
    end % protected Abstract method signatures
    
    methods      
        function obj = optimiseLamda( obj, Res, W, J, NumCovPar, MaxIter )
            %--------------------------------------------------------------
            % Optimise regularisation parameter for the current model
            % coefficients
            %
            % obj = obj.optimiseLamda( Lam, Res, W, J, NumCovPar, MaxIter );
            %
            % Input Arguments:
            %
            % Res       --> Residual vector
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % NumCovPar --> Number of covariance model parameters
            % MaxIter   --> Maximum number of iteration {25}. MaxIter is
            %               clipped to 1 as a minimum.
            %--------------------------------------------------------------
            if ( nargin < 7 )
                MaxIter = 25;                                               % Apply default
            elseif ( MaxIter < 1 )
                MaxIter = 1;                                                % Apply lower clip
            end
            %--------------------------------------------------------------
            % Compute starting value
            %--------------------------------------------------------------
            Lam = obj.initialLam( Res, W, J, NumCovPar );
            %--------------------------------------------------------------
            % Optimise the Lamda value
            %--------------------------------------------------------------
            obj = obj.reEstTemplate( Lam, Res,...
                W, J, NumCovPar, MaxIter );
        end
        
        function Lam0 = initialLam( obj, Res, W, J, NumCovPar, Int, Num )
            %--------------------------------------------------------------
            % Select best hyper-parameter in interval supplied.
            %
            % Lam0 = obj.initialLam( Res, W, J, NumCovPar, Int, Num );
            %
            % Input Arguments:
            %
            % Res       --> Residual vector
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % NumCovPar --> Number of covariance model parameters
            % Int       --> 1x2 vector of Lamda interval limits 
            %               {[0.00001, 1]}
            % Num       --> Number of samples for interval {6}
            %--------------------------------------------------------------
            if ( nargin < 6 )
                Int = [0.00001, 1];
            end
            if ( nargin < 7 )
                Num = 6;
            end
            Int = logspace( log10( min( Int ) ), log10( max( Int ) ), Num );
            DhDlam =  zeros( 1, Num );
            Ok = false( 1, Num );
            for Q = 1:Num
                %----------------------------------------------------------
                % Calculate absolute first derivative of h(lamda)
                %----------------------------------------------------------
                DhDlam( Q ) = abs( obj.firstDerivative( W, J,...
                    Res, Int( Q ), NumCovPar ) );
                %----------------------------------------------------------
                % Answer must be < 1 and in the interval
                %----------------------------------------------------------
                if ( ~isreal( DhDlam( Q ) ) )
                    DhDlam( Q ) = nan;
                end
                Ok( Q ) = ~isnan( DhDlam( Q ) );
                Ok( Q ) = Ok( Q ) & ( DhDlam( Q ) < 1 ) &...                          
                    ( DhDlam( Q ) >= min( Int ) ) &...
                    ( DhDlam( Q ) <= max( Int ) );
            end
            [ ~, Idx ] = min( DhDlam );
            Lam0 = Int( Idx );
            if ~Ok( Idx )
                %----------------------------------------------------------
                % Warn convergence is not guaranteed
                %----------------------------------------------------------
                warning( 'Initial Value for Lambda = %6.5e Value Not Guaranteed to Converge', Lam0 );
            end
        end
        
        function obj = reEstTemplate( obj, Lam, Res, W, J, NumCovPar, MaxIter )
            %--------------------------------------------------------------
            % Template for lamda re-estimation algorithm
            %
            % obj = obj.reEstTemplate( Lam, Res, W, J, NumCovPar, MaxIter );
            %
            % Input Arguments:
            %
            % Lam       --> Starting value for lamda re-estimation
            % Res       --> Residual vector
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % NumCovPar --> Number of covariance model parameters
            % MaxIter   --> Maximum number of iteration {25}. MaxIter is
            %               clipped to 1 as a minimum.
            %--------------------------------------------------------------
            if (nargin < 7) || isempty( MaxIter )
                MaxIter = 25;                                               % Apply default
            elseif MaxIter < 1
                MaxIter = 1;                                                % Must have at least one iteration
            end
            try
                obj.Lamda = obj.calculateLamda( Lam, Res, W,...
                    J, NumCovPar, MaxIter );                                % Update lamda
            catch
                obj.Lamda = Lam;
            end
            obj = obj.calcDoF( W, J );                                      % Update DoF
            obj = obj.getMeasure( obj.Lamda, Res, W, J, NumCovPar );        % Return the performance measure
        end
    
        function [S, Z, A, IA] = calcSmatrix( obj, Lam, W, J )
            %--------------------------------------------------------------
            % Calculate the S-matrix
            %
            % [S, Z, A, IA] = obj.calcSmatrix( Lam, W, J );
            %
            % Input Arguments:
            %
            % Lam   --> Regularisation parameter
            % W     --> Weight vector
            % J     --> Jacobean matrix
            %
            % Output Arguments:
            %
            % S     --> Smoother matrix Z*IA*Z.'
            % Z     --> J./sqrt(W)
            % A     --> (Z.'*Z + lam*I)
            % IA    --> inv( A )
            %--------------------------------------------------------------
            Z = obj.calcZmatrix( W, J );
            A = obj.calcAmatrix( Lam, Z );
            IA = A\eye( size( Z, 2 ) );
            S = Z*IA*Z.';
        end
        
        function S2 = calcSigma2( obj, Res, W, J, Lam )
            %--------------------------------------------------------------
            % Calculate the variance scale parameter
            %
            % S2 = obj.calcSigma2( Res, W, J, Lam );
            %
            % Input Arguments:
            %
            % Res       --> Residual vector
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % Lam       --> Regularisation coefficient {obj.Lamda}
            %--------------------------------------------------------------
            if ( nargin < 5 ) || isempty( Lam )
                Lam = obj.Lamda;                                            % Assign default
            end
            N = numel( Res );                                               % number of points
            C = sqrt( W );                                                  % Cholesky factor for W;
            Q = Res./C;                                                     % Weighted residual vector
            S = obj.calcSmatrix( Lam, W, J );                               % Return the necessary matrices
            S2 = Q.'*( eye( N ) - S )^2*Q/N;                                % Variance scale parameter
        end
        
        function obj = setLamda2Value( obj, Value )
            %--------------------------------------------------------------
            % Set regularisation parameter to desired value
            %
            % obj = obj.setLamda2Value( Value );
            %
            % Input Arguments:
            %
            % Value     --> Lamda >= 0
            %--------------------------------------------------------------
            obj.Lamda = Value;
        end
        
        function obj = calcDoF( obj, W, J, Lam )
            %--------------------------------------------------------------
            % Calculate the effective number of parameters
            %
            % obj = obj.calcDoF( W, J, Lam );
            %
            % Input Arguments:
            %
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % Lam       --> Regularisation coefficient {obj.Lamda}
            %--------------------------------------------------------------
            if ( nargin < 4 )
                Lam = obj.Lamda;
            end
            S = obj.calcSmatrix( Lam, W, J );
            obj.DoF = real ( trace( S ) );
        end
        
        function DHDL = firstDerivative( obj, W, J, Res, Lam, NumCovPar )
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
            H = 0.001*Lam;                                                  % Step size
            DHDL =  zeros( 1,5 );
            for Q = -2:2
                DLam = Lam + Q*H;                                           % Perturb the Lamda value
                DHDL( Q + 3 ) = obj.calcNewLam( W, J, Res, DLam, NumCovPar );
            end
            DHDL = (-DHDL(:,5) + 8*DHDL(:,4) - 8*DHDL(:,2) + DHDL(:,1))./(12*H);
        end
    end % constructor and ordinary methods
    
    methods ( Access = protected )   
        function [Lp, S] = calcProfileLikelihood( obj, Lam, Res, W, J )
            %--------------------------------------------------------------
            % Calculate the profile likelihood
            %
            % obj = obj.getMeasure( Lam, Res, W, J );
            %
            % Input Arguments:
            %
            % Lam       --> Starting value for lamda re-estimation
            % Res       --> Residual vector
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix
            %
            % Output Arguments:
            %
            % Lp        --> Profile likelihood
            % S         --> Smoother matrix
            %--------------------------------------------------------------
            N = numel( Res );
            S = obj.calcSmatrix( Lam, W, J );                               % Return the necessary matrices
            Q = Res./sqrt( W );
            S2 = Q.'*( eye( N ) - S )^2*Q/N;
            C = sqrt( W );
            Lp = 0.5*N*log( 2*pi ) + 0.5*N*log( S2 ) + sum( log( C ) ) + 0.5*N;
        end
    end % protected methods
    
    methods ( Access = private )
    end % private and helper methods
    
    methods
    end % get/set methods
    
    methods ( Static = true )
        function A = calcAmatrix( Lam, Z )
            %--------------------------------------------------------------
            % Calculate the A Matrix
            % 
            % A = RegFit.calcAmatrix( Lam, Z );
            % A = obj.calcAmatrix( Lam, Z );
            %
            % Input Arguments:
            %
            % Lam   --> Regularisation parameter
            % Z     --> Weighted Jacobean
            %--------------------------------------------------------------
            A = (Z.'*Z + Lam*eye( size( Z, 2 ) ) );
        end
        
        function Z = calcZmatrix( W, J )
            %--------------------------------------------------------------
            % Calculate the Z-matrix
            %
            % Z = RegFit.calcZmatrix( Lam, W, J );
            %
            % Input Arguments:
            %
            % W     --> Weight vector
            % J     --> Jacobean matrix
            %--------------------------------------------------------------
            C = sqrt( W );                                                  % Cholesky factor for weight matrix
            Z = diag(1./C)*J;                                               
        end
    end % static methods
end