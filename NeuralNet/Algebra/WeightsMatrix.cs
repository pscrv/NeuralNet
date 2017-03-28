using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class WeightsMatrix
    {
        #region private attributes
        private double[,] _matrix;
        #endregion

        #region public properties
        public int NumberOfOutputs { get { return _matrix.GetLength(0); } }
        public int NumberOfInputs { get { return _matrix.GetLength(1); } }        
        #endregion


        #region constructors
        public WeightsMatrix(double[,] matrix)
        {
            _matrix = (double[,])matrix.Clone();
        }

        public WeightsMatrix(int neurons, int inputs)
        {
            _matrix = new double[neurons, inputs];
        }
        #endregion


        #region public methods
        public void MultiplyBy(double factor)
        {
            for (int i = 0; i < NumberOfOutputs; i++)
                for (int j = 0; j < NumberOfInputs; j++)
                    _matrix[i, j] *= factor;
        }

        public void Add(WeightsMatrix other)
        {
            for (int i = 0; i < NumberOfOutputs; i++)
            {
                for (int j = 0; j < NumberOfInputs; j++)
                {
                    this._matrix[i, j] += other._matrix[i, j];
                }
            }
        }

        public void Subtract(WeightsMatrix other)
        {
            for (int i = 0; i < NumberOfOutputs; i++)
            {
                for (int j = 0; j < NumberOfInputs; j++)
                {
                    this._matrix[i, j] -= other._matrix[i, j];
                }
            }
        }

        public void Zero()
        {
            for (int i = 0; i < NumberOfOutputs; i++)
                for (int j = 0; j < NumberOfInputs; j++)
                    _matrix[i, j] = 0.0;
        }
        
        public NetworkVector LeftMultiply(NetworkVector vector)
        {
            double[] vectorarray = vector.ToArray();
            double sum;
            double[] result = new double[NumberOfOutputs];
            for (int i = 0; i < NumberOfOutputs; i++)
            {
                sum = 0;
                for (int j = 0; j < vector.Dimension; j++)
                {
                    sum += _matrix[i, j] * vectorarray[j];
                }
                result[i] = sum;
            }

            return new NetworkVector(result);
        }
        
        public NetworkVector DotWithWeightsPerInput(NetworkVector vector)
        {
            double[] result = new double[NumberOfInputs];
            NetworkVector inputWeights;
            for (int i = 0; i < NumberOfInputs; i++)
            {
                inputWeights = _getWeightsForOneInput(i);
                result[i] = vector.DotProduct(inputWeights);
            }
            
            return new NetworkVector(result);
        }

        public WeightsMatrix Copy()
        {
            return new WeightsMatrix(_matrix.Clone() as double[,]);
        }

        public double[,] ToArray()
        {
            return (double[,]) _matrix.Clone();
        }
        #endregion

        #region private methods
        private NetworkVector _getWeightsForOneInput(int inputindex)
        {
            double[] result = new double[NumberOfOutputs];
            for (int i = 0; i < NumberOfOutputs; i++)
            {
                result[i] = _matrix[i, inputindex];
            }
            return new NetworkVector(result);
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
