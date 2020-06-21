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
        D           int8        { mustBePositive( D ),...                   % Degree of interpolating polynomial vector
                                  mustBeFinite( D ), mustBeReal( D ),... 
                                  mustBeNonempty( D ),...
                                  mustBeInteger( D ) } = 3
        MetaData    struct                                                  % Metadata structure
    end % immutable properties    
    
    properties ( Constant = true )
        ModelName   RegFit.fitModelType = "tpss"                            % Model name
    end
    
    properties ( SetAccess = protected )
        ParNames    string                                                  % Parameter names
    end    
    
    properties ( SetAccess = protected, Dependent = true )
        K           double                                                  % Knot sequence
        Beta        double                                                  % Basis function coefficients
    end % protected properties    
    
    properties ( Dependent = true )
        Nb          int8                                                    % Number of basis functions
        NumFitCoeff int8                                                    % Total number of coefficients to be estimated from the data
    end % Dependent properties
    
    methods
        function obj = bspm( ReEstObj, Nk, D, MetaData )
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
            %               between knots. Must have obj.Nk + 2 elements.
            %               If a scalar then all polynomial
            %               degrees are the same in each segment. 
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
            % correct dimension (obj.Nk + 2)
            if isinteger( Value ) && ( numel( Value ) == 1 ) && ( Value > 0 )
                % set all segments to the same degree of polynomial.
                obj.D = Value*ones( obj.NumFitCoeff, 1 );                           %#ok<MCSUP>
            elseif all( isinteger( Value ) ) && ( numel( Value ) == ( obj.Nk + 2) ) %#ok<MCSUP>
                obj.D = Value( : );
            end
        end
    end % Get set methods
    
    methods ( Access = protected )
    end % protected methods
    
    methods ( Access = private )
        
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

