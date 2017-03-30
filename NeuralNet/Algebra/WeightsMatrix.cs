using System;
using System.Text;

using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet
{
    public class WeightsMatrix
    {
        #region private attributes
        private Matrix<double> _matrix;
        #endregion

        #region public properties
        public int NumberOfOutputs { get { return _matrix.RowCount; } }
        public int NumberOfInputs { get { return _matrix.ColumnCount; } }
        #endregion


        #region constructors
        public WeightsMatrix(double[,] matrix)
        {
            _matrix = Matrix<double>.Build.DenseOfArray(matrix);
        }

        public WeightsMatrix(int neurons, int inputs)
        {
            _matrix = Matrix<double>.Build.Dense(neurons, inputs);
        }

        public WeightsMatrix(Matrix<double> matrix)
        {
            _matrix = matrix;
        }
        #endregion


        #region public methods
        public void Scale(double factor)
        {
            _matrix = _matrix.Multiply(factor);
        }

        public void Add(WeightsMatrix other)
        {
            _matrix = _matrix.Add(other._matrix);
        }

        public void Subtract(WeightsMatrix other)
        {
            _matrix = _matrix.Subtract(other._matrix);
        }

        public void Zero()
        {
            _matrix.Clear();
        }
        
        public NetworkVector LeftMultiply(NetworkVector vector)
        {
            return new NetworkVector(_matrix.Multiply(vector.Vector));
        }
        
        public NetworkVector DotWithWeightsPerInput(NetworkVector vector)
        {
            Vector<double> result = Vector<double>.Build.Dense(_matrix.ColumnCount);

            foreach (Tuple<int, Vector<double>> index_column in _matrix.EnumerateColumnsIndexed())
            {
                result[index_column.Item1] = index_column.Item2.DotProduct(vector.Vector);
            }

            return new NetworkVector(result);

        }

        public WeightsMatrix Copy()
        {
            return new WeightsMatrix(_matrix);
        }

        public double[,] ToArray()
        {
            return _matrix.ToArray();
        }
        #endregion

        #region private methods
        private NetworkVector _getWeightsForOneInput(int inputindex)
        {
            return new NetworkVector( _matrix.Column(inputindex) );
        }
        #endregion


        #region overrides (comparison)
        public override bool Equals(object other)
        {
            if (ReferenceEquals(null, other))
                return false;

            if (ReferenceEquals(other, this))
                return true;

            if (other.GetType() != this.GetType())
                return false;

            return this.Equals(other as WeightsMatrix);
        }

        public bool Equals(WeightsMatrix other)
        {
            if (other == null)
                return false;

            if (this.NumberOfInputs != other.NumberOfInputs || this.NumberOfOutputs != other.NumberOfOutputs)
                return false;

            double epsilon = 0.000000001;
            for (int i = 0; i < this.NumberOfOutputs; i++)
                for (int j = 0; j < NumberOfInputs; j++)
                {
                    double difference = Math.Abs(this._matrix[i, j] - other._matrix[i, j]);
                    if (difference >= epsilon)
                        return false;
                }

            return true;
        }

        public override int GetHashCode()
        {
            int hash = 11;
            unchecked
            {
                for (int i = 0; i < NumberOfOutputs; i++)
                    for (int j = 0; j < NumberOfInputs; j++)
                    {
                        hash <<= 1;
                        hash ^= _matrix[i, j].GetHashCode();
                    }
            }
            return hash;
        }
        #endregion

        #region overrides
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < NumberOfOutputs; i++)
            {
                sb.Append("[");
                for (int j = 0; j < NumberOfInputs - 1; j++)
                {
                    sb.Append(_matrix[i, j].ToString());
                    sb.Append(",");
                }
                sb.Append(_matrix[i, NumberOfInputs - 1]);
                sb.Append("]");
            }
            return sb.ToString();
        }
        #endregion
    }
}
