using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class NetworkVector
    {
        #region delegates
        public delegate double SingleVariableFunction(double input);
        public delegate double TwoVariableFunction(double input1, double input2);
        #endregion

        #region private attributes
        private double[] _vector;
        #endregion

        #region public properties
        public int Dimension { get { return _vector.Length; } }

        // next line will not be needed, as methods are moved to this class
        // but perhaps needed as _matrix.clone()?
        public double[] Values
        {
            get { return (double[])_vector; }
            set { _vector = value; }
        }
        #endregion

        #region constructors
        public NetworkVector(double[] vector)
        {
            _vector = (double[])vector.Clone();
        }

        public NetworkVector(int dimensions)
        {
            _vector = new double[dimensions];
        }
        #endregion



        #region static methods
        public static NetworkVector ApplyFunctionComponentWise(NetworkVector vector, SingleVariableFunction fctn)
        {
            int dimension = vector.Dimension;
            double[] result = new double[dimension];
            for (int i = 0; i < dimension; i++)
            {
                result[i] = fctn(vector._vector[i]);
            }

            return new NetworkVector(result);
        }

        public static NetworkVector ApplyFunctionComponentWise(NetworkVector vector1, NetworkVector vector2, TwoVariableFunction fctn)
        {
            int dimension = vector1.Dimension;
            if (vector2.Dimension != dimension)
                throw new ArgumentException("Vector1 and Vector2 must have the same dimension.");


            double[] result = new double[dimension];
            for (int i = 0; i < dimension; i++)
            {
                result[i] = fctn(vector1._vector[i], vector2._vector[i]);
            }

            return new NetworkVector(result);
        }
        #endregion


        #region public methods
        public void SetValues(double[] values)
        {
            _vector = (double[]) values.Clone();
        }

        public NetworkVector SumWith(NetworkVector other)
        {
            if (this.Dimension != other.Dimension)
                throw new ArgumentException("Cannot add vectors of different dimention.");

            double[] result = new double[Dimension];
            for (int i = 0; i < Dimension; i++)
            {
                result[i] = other._vector[i] + this._vector[i];
            }

            return new NetworkVector(result);
        }

        public void Subtract(NetworkVector other)
        {
            if (other.Dimension != this.Dimension)
                throw new ArgumentException(string.Format("Attempt to subtract a vector of dimension {0} from a vector of dimsionsion {1}", other.Dimension, this.Dimension) );

            for (int i = 0; i < Dimension; i++)
            {
                this._vector[i] -= other._vector[i];
            }
        }

        public NetworkMatrix LeftMultiply(NetworkVector other)
        {
            double[,] result = new double[other.Dimension, this.Dimension];
            for (int i = 0; i < other.Dimension; i++)
            {
                for (int j = 0; j < this.Dimension; j++)
                {
                    result[i, j] = this._vector[j] * other._vector[i];
                }
            }
            return new NetworkMatrix(result);
        }

        public NetworkVector Copy()
        {
            return new NetworkVector(this._vector);
        }
        #endregion
    }
}
