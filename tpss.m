classdef tpss < RegFit.fitModel
    % One-dimensional truncated power series spline
    
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
        MetaData    struct                                                  % Metadata structure
    end % immutable properties    
    
    properties ( Constant = true )
        ModelName   RegFit.fitModelType = "tpss"                            % Model name
    end
    
    properties ( SetAccess = protected )
        ParNames    string                                                  % Parameter names
        Nb          int8                                                    % Number of basis functions
        D           int8        { mustBePositive( D ),...                   % Degree of interpolating polynomial vector
                                  mustBeFinite( D ), mustBeReal( D ),... 
                                  mustBeNonempty( D ),...
                                  mustBeInteger( D ) } = 2
    end % protected properties
    
    properties ( Access = private )
    end % Private properties
    
    properties ( SetAccess = protected, Dependent = true )
        K           double                                                  % Knot sequence
        Beta        double                                                  % Basis function coefficients
    end % protected properties    
    
    properties ( Dependent = true )
        NumFitCoeff int8                                                    % Total number of coefficients to be estimated from the data
    end % Dependent properties
    
    methods
        function obj = tpss( ReEstObj, Nk, D, MetaData )
            %--------------------------------------------------------------
            % class constructor
            %
            % obj = RegFit.tpss( ReEstObj, Nk, D, MetaData );
            %
            % Input Arguments:
            %
            % ReEstObj  --> RegFit.reEstLamdaContext object. Implements
            %               regularisation parameter re-estimation 
            %               algorithm.
            % Nk        --> Number of knots
            % D         --> Vector of degree of interpolating polynomials 
            %               between knots. Must have ( Nk + 1 ) elements.
            % MetaData  --> Custom structure containing meta data, e.g. 
            %               temperature, voltage, pressure,...
            %--------------------------------------------------------------
            obj.ReEstObj = ReEstObj;
            %--------------------------------------------------------------
            % Assign the knots
            %--------------------------------------------------------------
            obj.Nk = Nk;
            %--------------------------------------------------------------
            % Define the interpolating polynomial functions between the
            % knots
            %--------------------------------------------------------------
            obj.D = int8( D );
            %--------------------------------------------------------------
            % Assign auxilary data
            %--------------------------------------------------------------
            if ( nargin > 3 ) && isstruct( MetaData )
                obj.MetaData = MetaData;
            end
            %--------------------------------------------------------------
            % Calculate the number of basis functions
            %--------------------------------------------------------------
            obj.Nb = obj.calcNumBasis();
            %--------------------------------------------------------------
            % Set the bounds for the parameters
            %--------------------------------------------------------------
            obj = obj.setCoefficientBnds();
            %--------------------------------------------------------------
            % Set the parameter names
            %--------------------------------------------------------------
            obj = obj.setParameterNames();
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
            [ Knots, Coeff ] = obj.assignPars( Beta, obj.Nk );
            B = obj.basis( X, Knots );
            Yhat = B*Coeff;
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
            if ( nargin < 3 ) || isempty( Knots ) || ( numel( Knots ) ~= obj.Nk )
                Knots = obj.K;
            end
            %--------------------------------------------------------------
            % Augment the Knot sequence
            %--------------------------------------------------------------
            Knots = [ -inf; Knots( : ); inf ];
            %--------------------------------------------------------------
            % Create the basis function matrix
            %--------------------------------------------------------------
            B = zeros( numel( X ), ( obj.Nb - 1 ) );                             % Define storage
            Col = 1;
            for Q = 2:numel( Knots )
                Z = X;
                %----------------------------------------------------------
                % Define polynomial basis
                %----------------------------------------------------------
                if Col > obj.D( 1 )
                    %------------------------------------------------------
                    % Apply the knot adjustment
                    %------------------------------------------------------
                    P = ( X >= Knots( Q - 1 ) ) & ( X <= Knots( Q ) );
                    Z = Z - Knots( Q - 1 );
                    Z( ~P ) = 0;
                else
                end
                for J = 1:obj.D( Q - 1 )
                    B( :, Col ) = Z.^double( J );
                    Col = Col + 1;
                end
            end
            B = [ ones( size ( X ) ), B ];                                  % Add the intercept
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
            J = zeros( numel( X ), obj.NumFitCoeff );
            [ Knots, Coeff ] = obj.assignPars( Beta, obj.Nk );
            for Q = 1:obj.Nk
                J( :, Q ) = obj.diffBasisKnots( X, Knots )*Coeff;
            end
            J( :, (obj.Nk + 1):end ) = obj.basis( X, Knots );
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
            InitKnots = rand( Num, obj.Nk );                                % Random knot sequences
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
    end % constructor and ordinary methods
    
    methods
        function Beta = get.Beta( obj )
            % Retrieve basis function coefficients
            Beta = obj.Theta( ( obj.Nk + 1 ): end );
        end
        
        function K = get.K( obj )
            % Retrieve knot sequence
            K = obj.Theta( 1:obj.Nk );
        end
        
        function N = get.NumFitCoeff( obj )
            % Retrieve the total number of coefficients to be fitted
            N = obj.Nk + obj.Nb;
        end
        
        function obj = set.D( obj, Value )
            % Make sure the vector of interpolating polynomials has the
            % correct dimension (obj.Nk + 1)
            if all( isinteger( Value ) ) && ( numel( Value ) == ( obj.Nk + 1) ) %#ok<MCSUP>
                obj.D = int8( Value( : ) );
            else
                error('Property "D" not assigned');
            end
        end
    end % Get set methods
    
    methods ( Access = protected )
    end % protected methods
    
    methods ( Access = private )
        function DBDK = diffBasisKnots( obj, X, Knots )
            %--------------------------------------------------------------
            % Differentiate the basis function matrix wrt to the knot
            % positions.
            %
            % DBDK = obj.diffBasisKnots( X, Knots );
            %
            % Input Arguments:
            %
            % X     --> Data vector
            % Knots --> Knot vector in increasing order {obj.K}
            %--------------------------------------------------------------
            if ( nargin < 3 ) || isempty( Knots )
                Knots = obj.K;
            end
            DBDK = zeros( numel( X ), obj.Nb );
            C = obj.D( 1 ) + 1;                                             % Last column containing monomial for the first segment
            Knots = [ -inf; Knots( : ); inf ];                              % Augment the knot sequence
            for Q = 1:obj.Nk
                %----------------------------------------------------------
                % Differentiate the basis wrt the knot positions
                %----------------------------------------------------------
                Idx = 1:obj.D( Q + 1 );
                P = ( X >= Knots( Q ) ) & ( X <= Knots( Q + 1 ) );
                for J = Idx
                    C = C + 1;
                    DBDK( :, C ) = -double( J )*( X - Knots( Q ) ).^( double( J-1) );
                    DBDK( ~P, C ) = 0;
                end
            end
        end
        
        function obj = setCoefficientBnds( obj, LB, UB )
            %--------------------------------------------------------------
            % Set the coefficient and knot bounds 
            %
            % obj = obj.setCoefficientBnds();          % Use default bounds
            % obj = obj.setCoefficientBnds( LB, UB)    % Use custom bounds
            %
            % The default bounds are [-1, 1] for the knots, while the basis
            % function coefficients are undounded. These are highly
            % recommended.
            %
            % Input Arguments:
            %
            % LB    --> Lower bound vector for coefficients
            % UB    --> Upper bound vector for coefficients
            %--------------------------------------------------------------
            if ( nargin == 1 ) || ( numel( LB ) ~= ( obj.NumFitCoeff ) ) || ( numel( UB ) ~= ( obj.NumFitCoeff ) )
                LB = [ -ones( obj.Nk, 1 ); -inf( obj.Nb, 1 ) ];
                UB = [ ones( obj.Nk, 1 ); inf( obj.Nb, 1 ) ];
            end
            obj.LB = LB( : );
            obj.UB = UB( : );
        end
        
        function Nb = calcNumBasis( obj )
            %--------------------------------------------------------------
            % Calculate the number of basis functions
            %
            % Nb = obj.calcNumBasis();
            %--------------------------------------------------------------
            Nb = sum( obj.D ) + 1;
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
                Pars( Q ) = string( [ 'b_', num2str( Q - obj.Nk ) ] );
            end
            obj.ParNames = Pars;
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
    end % static methods
end

