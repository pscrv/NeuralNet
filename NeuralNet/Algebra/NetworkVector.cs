﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class NetworkVector : IEquatable<NetworkVector>
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
        public static NetworkVector Sum(IEnumerable<NetworkVector> vectors)
        {
            int dimension = vectors.ElementAt(0).Dimension;
            if (vectors.Any(x => x.Dimension != dimension))
                throw new ArgumentException("Attempt do add vectors of different sizes.");
            double[] result = new double[dimension];
            foreach (NetworkVector vector in vectors)
            {
                for (int i = 0; i < dimension; i++)
                {
                    result[i] += vector._vector[i];
                }
            }
            return new NetworkVector(result);
        }

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

        public static NetworkVector Concatenate(IEnumerable<NetworkVector> vectorsToConcatenate)
        {
            if (vectorsToConcatenate == null || vectorsToConcatenate.Count() == 0)
                throw new ArgumentException("Attempt to concatenate null or empty IEnumerable<NetworkVector.");
            
            int totalDimension = vectorsToConcatenate.Sum(x => x.Dimension);

            double[] resultVector = new double[totalDimension];
            int index = 0;
            foreach (NetworkVector vector in vectorsToConcatenate)
            {
                Array.Copy(vector._vector, 0, resultVector, index, vector.Dimension);
                index += vector.Dimension;
            }
            return new NetworkVector(resultVector);
        }
        #endregion


        #region public methods
        public double SumValues()
        {
            double sum = 0.0;
            for (int i = 0; i < Dimension; i++)
            {
                sum += _vector[i];
            }

            return sum;
        }

        public NetworkVector SumWith(NetworkVector other)
        {
            if (this.Dimension != other.Dimension)
                throw new ArgumentException("Cannot add vectors of different dimension.");

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

        public double DotProduct(NetworkVector other)
        {
            if (this.Dimension != other.Dimension)
                throw new ArgumentNullException("Attempt to form dot product, but dimensions do not match.");
            double sum = 0.0;
            for (int i = 0; i < Dimension; i++)
            {
                sum += this._vector[i] * other._vector[i];
            }
            return sum;
        }

        public NetworkMatrix LeftMultiply(NetworkVector other)
        {
            double[,] result = new double[this.Dimension, other.Dimension];
            for (int i = 0; i < this.Dimension; i++)
            {
                for (int j = 0; j < other.Dimension; j++)
                {
                    result[i, j] = this._vector[i] * other._vector[j];
                }
            }
            return new NetworkMatrix(result);
        }

        public List<NetworkVector> Segment(int partCount)
        {
            if (partCount <= 0)
                throw new ArgumentException("Attempt to segment into fewer than one part.");

            if (Dimension % partCount != 0)  // drop this and rely on the caller, for speed?
                throw new ArgumentException("Attempt to segment a NetworkVector into unequal parts.");

            if (partCount == 1)
                return new List<NetworkVector> { this };

            int partDimension = Dimension / partCount;
            double[] part = new double[partDimension];
            List<NetworkVector> result = new List<NetworkVector>();

            for (int i = 0; i < partCount; i++)
            {
                Array.Copy(_vector, i * partDimension, part, 0, partDimension);
                result.Add(new NetworkVector(part));
            }
            return result;
        }


        public NetworkVector Copy()
        {
            return new NetworkVector(this._vector.Clone() as double[]);
        }

        public double[] ToArray()
        {
            return (double[]) _vector.Clone();
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

            return this.Equals(other as NetworkVector);
        }

        public bool Equals(NetworkVector other)
        {
            if (other == null)
                return false;

            if (this.Dimension != other.Dimension)
                return false;

            double epsilon = 0.000000001;
            for (int i = 0; i < this.Dimension; i++)
            {
                double difference = Math.Abs(this._vector[i] - other._vector[i]);
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
                for (int i = 0; i < Dimension; i++)
                {
                    hash <<= 1;
                    hash ^= _vector[i].GetHashCode();
                }
            }
            return hash;
        }
        #endregion

        #region overrides
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append ("[");
            for (int i = 0; i < Dimension- 1; i++)
            {
                sb.Append(_vector[i].ToString());
                sb.Append(",");
            }
            sb.Append(_vector[Dimension].ToString());
            return sb.ToString();
        }
        #endregion
    }
}
