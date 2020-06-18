classdef reEstLamdaContext
    % Optimal hyper-parameter estimation user interface class
    
    properties ( SetAccess = protected )
        ReEstObj    {mustBeReEstModel( ReEstObj )}                          % reEstLamda object
    end
    
    properties ( SetAccess = protected, Dependent = true )
        Lamda                                                               % Regularisation parameter
        DoF                                                                 % Model degree of freedom including covariance parameters
        Algorithm                                                           % Re-estimation algorithm name
        Measure                                                             % Information theoretic performance measure
    end
    
    methods
        function obj = reEstLamdaContext( ReEstObj )
            %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.reEstLamdaContext( ReEstObj );
            %
            % Input Arguments:
            %
            % ReEstObj  --> RegFit.reEstLamda object
            %--------------------------------------------------------------
            obj.ReEstObj = ReEstObj;
        end
        
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
            obj.ReEstObj = obj.ReEstObj.reEstTemplate( Lam, Res,...
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
                DhDlam( Q ) = abs( obj.ReEstObj.firstDerivative( W, J,...
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
            obj.ReEstObj = obj.ReEstObj.getMeasure( Lam, Res,...
                W, J, NumCovPar );
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
            [S, Z, A, IA] = obj.ReEstObj.calcSmatrix( Lam, W, J );
        end
        
        function S2 = calcSigma2( obj, Res, W, J, Lam )
            %--------------------------------------------------------------
            % Calculate the variance scale parameter
            %
            % S2 = obj.calcSigma2( Res, W, J, Lam );
            % Res       --> Residual vector
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % Lam       --> Regularisation coefficient {obj.Lamda}
            %-------------------------------------------------------------- 
            if ( nargin < 5 )
                Lam = obj.Lamda;                                            % Apply default
            end
            S2 = obj.ReEstObj.calcSigma2( Res, W, J, Lam );
        end
        
        function obj = calcDoF( obj, W, J, Lam )
            %--------------------------------------------------------------
            % Calculate the effective number of parameters
            %
            % obj = obj.callcDoF( W, J, Lam );
            %
            % Input Arguments:
            %
            % W         --> Weight vector (including multiplication by
            %               variance scale parameter)
            % J         --> Jacobean matrix 
            % Lam       --> Regularisation coefficient {obj.Lamda}
            %--------------------------------------------------------------    
            if ( nargin < 4 )
                Lam = obj.Lamda;                                            % Apply default
            end
            obj = obj.ReEstObj.calcDoF( W, J, Lam );
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
            obj.ReEstObj = obj.ReEstObj.setLamda2Value( Value );
        end
    end % constructor and ordinary methods
    
    methods
        function L = get.Lamda( obj )
            % Return regularisation parameter
            L = obj.ReEstObj.Lamda;
        end
        
        function D = get.DoF( obj )
            % Return degrees of freedom
            D = obj.ReEstObj.DoF;
        end
        
        function M = get.Measure( obj )
            % Return information theoretic performance measure
            M = obj.ReEstObj.Measure;
        end
        
        function A = get.Algorithm( obj )
            % Return re-estimation algorithm name
            A = obj.ReEstObj.Name;
        end
    end % get/set methods
    
    methods
    end % private and helper methods
end

function mustBeReEstModel( ReEstObj )
    %----------------------------------------------------------------------
    % Validator function for ReEstObj property
    %
    % mustBeReEstModel( ReEstObj )
    %----------------------------------------------------------------------
    Names = ["aicReEst", "bicReEst"];
    if ~isempty( ReEstObj ) && ~contains( ReEstObj.Name, Names ) 
        error('Unrecognised re-estimation method option');
    end
end