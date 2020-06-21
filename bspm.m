classdef bspm < RegFit.fitModel
    % One-dimensional B-spline model
    
    properties 
        Theta       double                                                  % Parameter vector
        LB          double                                                  % Lower bound constraint for knots
        UB          double                                                  % Upper bound constraint for knots
    end % public properties
    
    properties( SetAccess = immutable )
        Nk          int8        { mustBePositive( Nk ),...                  % Number of knots
                                  mustBeFinite( Nk ), mustBeReal( Nk ),... 
                                  mustBeNonempty( Nk ),...
                                  mustBeInteger( Nk ) } = 1                                              
        D           int8        { mustBePositive( D ),...                   % Degree of interpolating polynomial
                                  mustBeFinite( D ), mustBeReal( D ),... 
                                  mustBeNonempty( D ),...
                                  mustBeInteger( D ) } = 3
        MetaData    struct                                                  % Metadata structure
    end % immutable properties
    
    properties ( SetAccess = protected, Dependent = true )
        K           double                                                  % Knot sequence
        Beta        double                                                  % Basis function coefficients
    end % protected properties
    
    properties ( Dependent = true )
        M           int8                                                    % Bspline order
        Nb          int8                                                    % Number of basis functions
        NumFitCoeff int8                                                    % Total number of coefficients to be estimated from the data
    end % Dependent properties
    
    properties ( Access = private, Dependent = true )
    end % Dependent & private properties
    
    properties ( SetAccess = protected )
        ParNames    string                                                  % Parameter names
    end
    
    properties ( Constant = true )
        ModelName   RegFit.fitModelType = "BSPL"                            % Model name
    end % Constant properties
    
    methods
        function obj = bspm( ReEstObj, Nk, D, MetaData )
            %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.bspm( ReEstObj, Nk, D, MetaData );
            %
            % Input Arguments:
            %
            % ReEstObj  --> RegFit.reEstLamdaContext object. Implements
            %               regularisation parameter re-estimation 
            %               algorithm.
            % Nk        --> Number of knots
            % D         --> Degree of interpolating polynomial
            % MetaData  --> Custom structure containing meta data, e.g. 
            %               temperature, voltage, pressure,...
            %--------------------------------------------------------------
            obj.ReEstObj = ReEstObj;
            obj.Nk = Nk; 
            obj.D = D;
            %--------------------------------------------------------------
            % Assign auxilary data
            %--------------------------------------------------------------
            if ( nargin > 3 )
                obj.MetaData = MetaData;
            end
            %--------------------------------------------------------------
            % Set the bounds for the parameters
            %--------------------------------------------------------------
            obj = obj.setCoefficientBnds();
            %--------------------------------------------------------------
            % Set the parameter names
            %--------------------------------------------------------------
            obj = obj.setParameterNames();
        end
        
        function B = basis( obj, X, Knots )
            %--------------------------------------------------------------
            % Return basis function matrix for supplied knot sequence
            %
            % B = obj.basis( X, Knots );
            %
            % Input Arguments:
            %
            % X     --> Input data
            % Knots --> Knot sequence { obj.K }
            %--------------------------------------------------------------
            if ( nargin < 3 ) || isempty( Knots )
                Knots = obj.K;
            end
            AugKnotSequence = obj.augKnots( Knots );
            B = obj.phi_calc( AugKnotSequence, obj.D, X(:) );
        end
        
        function J = jacobean( obj, X, Beta )    
            %--------------------------------------------------------------
            % Return Jacobean matrix for the supplied parameter vector
            %
            % J = obj.jacobean( X, Beta );
            %
            % Input Arguments:
            %
            % X     --> Input data clipped to range [0, 1];
            % Beta  --> Vector of parameters { obj.Theta }
            %--------------------------------------------------------------
            if ( nargin < 3 ) || isempty( Beta )
                Beta = obj.Theta;
            end
            [ Knots, Coeff ] = obj.assignPars( Beta, obj.Nk );
            J = zeros( numel( X ), obj.Nk + obj.Nb );
            %--------------------------------------------------------------
            % Calculate derivative with respect to the knots using the
            % 5-point stencil formula with dK( Q ) = 0.0001 [%]
            %--------------------------------------------------------------
            J( :, 1:obj.Nk ) = obj.dBdK( Knots, Coeff, X, 0.0001 );
            J( :, ( obj.Nk + 1 ):end ) = obj.basis( X, Knots );
        end
        
        function Yhat = predictions( obj, X, Beta )
            %--------------------------------------------------------------
            % Model predictions for the supplied parameter vector.
            %
            % Yhat = obj.predictions( X, Beta );
            %
            % Input Arguments:
            %
            % X     --> Input data clipped to range [0, 1];
            % Beta  --> Vector of parameters { obj.Theta }
            %--------------------------------------------------------------
            if ( nargin < 3 ) || isempty( Beta )
                Beta = obj.Theta;
            end
            [ Knots, Beta ] = obj.assignPars( Beta, obj.Nk );
            B = obj.basis( X, Knots );
            Yhat = B*Beta;
        end
        
        function V = startingValues( obj, X, Y, Num )
            %--------------------------------------------------------------
            % Generate starting values for the knot positions and basis
            % function coefficients
            %
            % obj = obj.startingValues( X, Y, Num );
            %
            % Input Arguments:
            %
            % X     --> Input data
            % Y     --> Observed response data
            % Num   --> Number of random knot locations {25}
            %--------------------------------------------------------------
            if ( nargin < 4 ) || isempty( Num )
                Num = 25;                                                   % Apply default
            end
            InitKnots = rand( Num, obj.Nk );                               % Random knot sequences
            InitKnots = sort( InitKnots, 2 );                               % Strictly increasing knots
            %--------------------------------------------------------------
            % Clip the initial random knots to (0, 1)
            %--------------------------------------------------------------
            P = any( ( InitKnots > 0 ) & ( InitKnots < 1 ), 2 );
            InitKnots = InitKnots( P, : );
            InitKnots = InitKnots.';
            Num = sum( P );
            L = nan( Num, 1 );
            Coeff = cell( Num, 1 );
            %--------------------------------------------------------------
            % Remove data with inifinite weights and input outside interval
            % ( 0, 1 )
            %--------------------------------------------------------------
            W = ones( size( X ) );
            [X, Y, W] = obj.parseInputs( X, Y, W );
            for Q = 1:Num
                %----------------------------------------------------------
                % Evaluate cost function for each random knot combination
                % and choose the one with the best result
                %----------------------------------------------------------
                Knots = InitKnots( :, Q );                                  % Assign knots
                B = obj.basis( X, Knots );
                Z = B.'*B + obj.Lamda*eye( obj.Nb );
                Z = Z\eye( obj.Nb );
                Coeff{ Q } = Z*B.'*Y;                                       % Compute current basis fcn coefficients
                V = [ Knots; Coeff{ Q } ];
                L( Q ) = obj.costFcn( V, X, Y, W, 1 ); 
            end
            [ ~, Idx ] = min( L );
            V = [ InitKnots( :, Idx ); Coeff{ Idx } ]; 
        end
    end % ordinary and constructor methods
    
    methods
        function obj = set.Theta( obj, Value )
            % Set the knot sequence
            Ok = numel( Value ) == obj.NumFitCoeff;             %#ok<MCSUP> % % Number of parameters must be correct
            Ok = Ok & all( Value( 1:obj.Nk ) > obj.LB( 1 ) );   %#ok<MCSUP> % Knots must be greater than lower bound for data
            Ok = Ok & all( Value( 1:obj.Nk ) < obj.UB( 1 ) );   %#ok<MCSUP> % Knots must be less than upper bound for data
            if Ok
                Value( 1:obj.Nk ) = sort(...
                    Value( 1:obj.Nk ) );                        %#ok<MCSUP> % Ensure strictly increasing knots
                obj.Theta = Value; 
            else
                warning('Parameters not assigned');
            end
        end
        
        function K = get.K( obj )
            % Retrieve knot sequence
            K = obj.Theta( 1:obj.Nk );
        end
        
        function Beta = get.Beta( obj )
            % Retrieve basis function coefficients
            Beta = obj.Theta( ( obj.Nk + 1 ): end );
        end
        
        function M = get.M( obj )
            % Retrieve spline order as an integer
            M = obj.D + 1;
        end
        
        function Nb = get.Nb( obj )
            % Retrieve number of basis functions
            Nb = obj.M + obj.Nk;
        end       
        
        function N = get.NumFitCoeff( obj )
            % Retrieve the total number of coefficients to be fitted
            N = obj.Nk + obj.Nb;
        end
    end % Get/Set methods
    
    methods ( Access = protected )
        function C = mleConstraints( obj, Beta ) 
            %--------------------------------------------------------------
            % Provide custom constraints for optimisation. See help for
            % fmincon for definitions.
            %
            % Input Arguments:
            %
            % Beta  --> Coefficient vector. Decision variables for
            %           optimisation of the cost function for RIGLS
            %
            % Output Arguments:
            %
            % C     --> Structure of constraints with fields:
            %           Aineq       --> Linear inequality constraint
            %                           coefficient matrix.
            %           bineq       --> Linear inequality constraints bound
            %                           matrix.
            %           Aeq         --> Linear equality constraint
            %                           coefficient matrix.
            %           beq         --> Linear equality constraints bound
            %                           matrix.
            %           nonlcon     --> Nonlinear constraints function
            %--------------------------------------------------------------
            [ C.Aineq, C.bineq ] = obj.genIneqCon( Beta );
            C.Aeq = [];
            C.beq = [];
            C.nonlcon = [];
        end
    end
    
    methods ( Access = private )
        function [ Aineq, Bineq ] = genIneqCon( obj, Beta )                 %#ok<INUSD>
            %--------------------------------------------------------------
            % Generate inequality constraints for knots
            %
            % [ Aineq, Bineq ] = obj.genIneqCon( Beta );
            %
            % Input Arguments:
            %
            % Beta  --> Coefficient vector. Decision variables for
            %           optimisation of the cost function for RIGLS
            %
            % Output Arguments:
            %
            % Aineq --> Linear inequality constraint coefficient matrix.
            % bineq --> Linear inequality constraints bound matrix.
            %--------------------------------------------------------------
            Aineq = []; 
            Bineq = [];
            if ( obj.Nk > 1 )
                %----------------------------------------------------------
                % Apply the knot minimum separation constraints
                %
                % Aineq*Beta <= Bineq
                %----------------------------------------------------------
                Aineq = zeros( ( obj.Nk - 1 ), ( obj.NumFitCoeff ) );
                for Q = 1:( obj.Nk - 1 )
                    %----------------------------------------------------------
                    % Knots are in strictly increasing order. Ensure difference
                    % is greater than 0.05 on the coded scale.
                    %----------------------------------------------------------
                    Aineq( Q, Q:( Q + 1 ) ) = [ 1, -1 ];
                end
                %--------------------------------------------------------------
                % Note this relies on the x-data interval being [0, 1].
                %
                % TODO: make this more generic!!!!!
                %--------------------------------------------------------------
                Bineq = -0.05*ones( ( obj.Nk - 1 ), 1 );
            end
        end
        
        function obj = setParameterNames( obj )
            %--------------------------------------------------------------
            % Set the coefficient and knot names
            %
            % obj = obj.setParameterNames();
            %--------------------------------------------------------------
            Pars = string.empty( 0, obj.NumFitCoeff );
            for Q = 1:obj.Nk
                %----------------------------------------------------------
                % Define the knots
                %----------------------------------------------------------
                Pars( Q ) = string( [ 'k_', num2str( Q ) ] );       
            end
            for Q = ( obj.Nk + 1 ):obj.NumFitCoeff
                %----------------------------------------------------------
                % Assign the basis function coefficient names
                %----------------------------------------------------------
                Pars( Q ) = string( [ '\beta_', num2str( Q - obj.Nk ) ] );
            end
            obj.ParNames = Pars;
        end
        
        function obj = setCoefficientBnds( obj, LB, UB )
            %--------------------------------------------------------------
            % Set the coefficient and knot bounds 
            %
            % obj = obj.setCoefficientBnds();          % Use default bounds
            % obj = obj.setCoefficientBnds( LB, UB)    % Use custom bounds
            %
            % The default bounds are [0, 1] for the knots, while the basis
            % function coefficients are undounded. These are highly
            % recommended.
            %
            % Input Arguments:
            %
            % LB    --> Lower bound vector for coefficients
            % UB    --> Upper bound vector for coefficients
            %--------------------------------------------------------------
            if ( nargin == 1 ) || ( numel( LB ) ~= ( obj.Nk + obj.Nb ) ) || ( numel( UB ) ~= ( obj.Nk + obj.Nb ) )
                LB = [ zeros( obj.Nk, 1 ); -inf( obj.Nb, 1 ) ];
                UB = [ ones( obj.Nk, 1 ); inf( obj.Nb, 1 ) ];
            end
            obj.LB = LB( : );
            obj.UB = UB( : );
        end
        
        function Aug = augKnots( obj, Knots )
            %--------------------------------------------------------------
            % Augment the supplied knots sequence for basis function
            % calculations. Add obj.M knots for obj.LB( 1 ) to start of 
            % knot sequence and obj.M knots for obj.UB( 1 ) to end of knot 
            % sequence.
            %
            % Aug = obj.augKnots( Knots );
            %
            % Input Arguments:
            %
            % Knots --> Knot sequence { obj.K }
            %--------------------------------------------------------------
            if ( nargin < 2 )
                Knots = obj.K;
            end
            Knots = sort( Knots );
            Aug = [ repmat( obj.LB( 1 ), obj.M, 1 ); Knots(:);...
                repmat( obj.UB( 1 ), obj.M, 1 ) ];
        end
        
        function Db = diffBasis(obj, X, R, Knots)
            %--------------------------------------------------------------
            % Calculate the rth derivative of the basis functions for the
            % current knot sequence
            %
            % Db = obj.diffBasis( X, R, Knots )
            %
            % Input Arguments:
            %
            % X     --> vector of design sites
            % R     --> order of the derivative {1}
            % Knots --> Knot sequence { obj.K }
            %--------------------------------------------------------------
            X = X(:);
            if ( nargin < 3 )
                R = 1; % default derivative
            elseif ( R <= obj.D )
                R = round( R ); % user supplied derivative
            else
                error('R must be in the interval [1, %3.0d]', obj.D );
            end
            if ( nargin < 4 ) || isempty( Knots )
                Knots = obj.K;                                              % Applydefault
            end
            Aug = obj.augKnots( Knots );                                    % Generates augmented knot sequence
            Db = myBspline.phi_calc( Aug, ( obj.D - R ) , X);               % Calculate the (m-r)th basis functions
            for Q=1:R
                %----------------------------------------------------------
                % Recursively calculate the derivative matrices H and L
                %----------------------------------------------------------
                L = -eye( obj.Nb - Q ) + diag( ones( obj.Nb - Q -1, 1 ),...
                    1 );
                L = [L [zeros( obj.Nb - Q - 1, 1 ); 1 ] ];                  %#ok<AGROW>
                
                H = zeros( obj.Nb - Q, 1 );
                
                for T = ( Q + 1 ):obj.Nb
                    H( T - Q ) = 0.5*( obj.M - Q )/...
                        ( Aug( T + obj.m - Q ) - Aug(T));
                end
                H = diag( H );
                if ( Q < 2 )
                    B = H*L;
                else
                    B = H*L*B;
                end
            end
            %--------------------------------------------------------------
            % calculate basis for the spline derivative
            %--------------------------------------------------------------
            Db =Db*B;  
        end
        
        function Dy = calcDerivative( obj, X, R, Parameters )
            %--------------------------------------------------------------
            % Calculate the rth derivative of the B-spline for the
            % current knot sequence
            %
            % Db = obj.calcDerivative( X, R, Parameters )
            %
            % Input Arguments:
            %
            % X             --> vector of design sites
            % R             --> order of the derivative {1}
            % Parameters    --> Knot sequence { obj.Theta }
            %--------------------------------------------------------------
            X = X(:);
            if ( nargin < 3 )
                R = 1; % default derivative
            elseif ( R <= obj.D )
                R = round( R ); % user supplied derivative
            else
                error('R must be in the interval [1, %3.0d]', obj.D );
            end
            if ( nargin < 4 ) || isempty( Parameters )
                Parameters = obj.Theta;                                     % Applydefault
            end
            [ Knots, Coeff ] = obj.assignPars( Parameters );
            B = obj.diffBasis(X, R, Knots);                                 % Differentiate the basis
            Dy = B*Coeff;
        end
        
        function Df = dBdK( obj, Knots, Coeff, X, Del )
            %--------------------------------------------------------------
            % Derivative of the basis with respect to the knots
            %
            % Df = obj.dBdK( Knots, Coeff, X, Del );
            %
            % Input Arguments:
            %
            % Knots     --> Knot vector { obj.K }
            % Coeff     --> Basis function coefficient vector { obj.Beta }
            % X         --> Data points to evaluate the basis at
            % Del       --> Knot perturbation percentage { 0.0001 }
            %
            % Output Arguments:
            %
            % Df        --> N by obj.Nk matrix of derivatives where N is
            %               the number of data points
            %--------------------------------------------------------------
            if ( nargin < 5 )
                Del = 0.0001;                                               % Apply default
            end
            Del = Knots*Del;                                                % Apply percentage perturbation
            Df = zeros( numel( X ), obj.Nk );                               % Assign storage
            for Q = 1:obj.Nk
                %----------------------------------------------------------
                % Calculate the derivatives using the 5-point stencil
                %----------------------------------------------------------
                F = zeros( numel( X ), 5 );
                for S = -2:2
                    Dk = zeros( size( Knots ) );
                    Dk( Q ) = S*Del( Q );
                    Dknot = Knots + Dk;                                     % Perturb the knot values
                    F( :, S+3 ) = obj.basis( X, Dknot )*Coeff;              % F( Knots( Q ) + S*Del ) 
                end
                F = fliplr( F );
                Df( :, Q ) = ( -F( :, 1 ) + 8*F( :, 2 ) + 0*F( :, 3 ) -...
                    8*F( :, 4 ) + F( :, 5 ) )./ Del( Q ) / 12;
            end
        end
    end % private and helper methods
    
    methods ( Static = true )
        function [X, Y, W] = parseInputs( X, Y, W )
            %--------------------------------------------------------------
            % Remove zero W as it causes infinite weights and eliminate
            % data outside range [0, 1]
            %
            % [X, Y, W] = RegFit.bspm.parseInputs( X, Y, W );
            % [X, Y, W] = obj.parseInputs( X, Y, W );
            %
            % X             --> Independent data
            % Y             --> Observed data vector
            % W             --> Weights
            %--------------------------------------------------------------
            P = ( X < 0 ) | ( X > 1 ) | ( W == 0 );
            X = X( ~P );
            Y = Y( ~P );
            W = W( ~P );
        end
        
        function [ K, Coeff ] = assignPars( Beta, Nk )
            %--------------------------------------------------------------
            % Assign the parameter vector contents
            %
            % First Nk terms are the knots, the remainder the basis
            % function coefficients.
            %
            % Sorts the current knot sequence so it is strictly increasing
            %--------------------------------------------------------------
            K = sort( Beta( 1:Nk ) );
            Coeff = Beta( ( Nk + 1 ):end );
        end
        
        function PHI = phi_calc( knots, s, X)
            %--------------------------------------------------------------
            % Calculate B-spline basis
            %
            % PHI= PHI_CALC(knots,s,x);
            %
            % Input Arguments:
            %
            %  	knots    a vector of augmented knot positions. Note the 
            %            outer knots [a, b] must be repeated s+1 times
            %	  s      the order of the spline
            %	  x      is a vector of xvalues that are to be evaluated.
            %
            % Output Arguments:
            %
            %   PHI    the matrix of phi values
            %
            %
            %  Copyright 2000-2004 The MathWorks, Inc. and Ford Global Technologies, Inc.
            %   $Revision: 1.4.4.2 $  $Date: 2004/02/09 07:44:18 $
            %
            % COPIED FROM MBC TOOLBOX....
            %--------------------------------------------------------------
            
            %DEFINE VARIABLES
            os=s+1; 				  %offset for matrix referencing
            k=length(knots)-2*os; 	%number of knots
            
            %SET UP THE KNOT POSITIONS
            K= knots(:);
            
            B1= zeros(length(X),2*s+k);
            for i= os:os+k
                % extrapolate below -1
                B1(X<K(1), i)   = K(i) <= K(1);
                % extrapolate above +1
                B1(X>=K(end), i) = K(i+1)==K(end);
                % interpolate
                B1(( K(i) <= X) & (X < K(i+1) ),i)= 1;
            end
            
            %RECURSIVE SECTION
            %loop through the levels
            for j = 2 : s+1
                % save last level
                B0=B1;
                for i= 1:2*s+k+2-j  % matrix of index points
                    dK= K(i+j-1)-K(i);
                    if dK~=0
                        B1(:,i)= ((X-K(i))/dK).* B0(:,i);
                    else
                        B1(:,i)= 0;
                    end
                    
                    dK=K(i+j)-K(i+1);
                    if dK~=0
                        B1(:,i)= B1(:,i) + ((K(i+j)-X)/dK) .* B0(:,i+1);
                    end
                    
                end
            end
            
            % only need the first s+k+1 columns
            PHI= B1(:,1:s+k+1);
        end
    end % Static methods
end