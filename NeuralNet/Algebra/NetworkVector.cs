using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet
{
   

    public class NetworkVector : IEquatable<NetworkVector>
    {
        #region delegates
        public delegate double SingleVariableFunction(double input);
        public delegate double TwoVariableFunction(double input1, double input2);
        #endregion

        #region private attributes
        protected Vector<double> _vector;
        #endregion

        #region public properties
        public int Dimension { get { return _vector.Count; } }
        public Vector<double> Vector { get { return _vector; } }
        #endregion

        #region constructors
        protected NetworkVector() { }

        public NetworkVector(double[] vector)
        {
            _vector = Vector<double>.Build.DenseOfArray(vector);
        }

        public NetworkVector(IEnumerable<double> vector)
        {
            _vector = Vector<double>.Build.DenseOfEnumerable(vector);
        }

        public NetworkVector(int dimension)
        {
            _vector = Vector<double>.Build.Dense(dimension);
        }

        public NetworkVector(Vector<double> vector)
        {
            _vector = Vector<double>.Build.DenseOfVector(vector);
        }
        #endregion



        #region static methods
        public static NetworkVector Sum(IEnumerable<NetworkVector> vectors)
        {
            return new NetworkVector( vectors.Select(x => x._vector).Aggregate((x, y) => x + y) );
        }

        public static NetworkVector ApplyFunctionComponentWise(NetworkVector vector, SingleVariableFunction fctn)
        {
            return new NetworkVector( vector._vector.Select(x => fctn(x)) );            
        }

        public static NetworkVector ApplyFunctionComponentWise(NetworkVector vector1, NetworkVector vector2, TwoVariableFunction fctn)
        {
            return new NetworkVector(vector1._vector.Zip(vector2._vector, (x, y) => fctn(x, y)));
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
                Array.Copy(vector._vector.ToArray(), 0, resultVector, index, vector.Dimension);
                index += vector.Dimension;
            }
            return new NetworkVector(resultVector);
        }
        #endregion


        #region public methods
        public void Zero()
        {
            _vector.Clear();
        }

        public void Subtract(NetworkVector other)
        {
            _vector = _vector.Subtract(other._vector);
        }

        public void Add(NetworkVector other)
        {
            _vector = _vector.Add(other._vector);
        }

        public void Scale(double factor)
        {
            _vector = _vector.Multiply(factor);
        }


        public NetworkVector SumWith(NetworkVector other)
        {
            return new NetworkVector( _vector.Add(other._vector));
        }

        public WeightsMatrix LeftMultiply(NetworkVector other)
        {
            return new WeightsMatrix(_vector.OuterProduct(other._vector));
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
            Vector<double> part;
            List<NetworkVector> result = new List<NetworkVector>();

            for (int i = 0; i < partCount; i++)
            {
                part = _vector.SubVector(i * partDimension, partDimension);
                result.Add(new NetworkVector(part));
            }

            return result;
        }

        public NetworkVector Copy()
        {
            return new NetworkVector(_vector);
        }


        public double SumValues()
        {
            return _vector.Sum();
        }

        public double DotProduct(NetworkVector other)
        {
            return _vector.DotProduct(other._vector);
        }


        public double[] ToArray()
        {
            return _vector.ToArray();
        }
        #endregion


        #region IEquatable
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
            sb.Append("[");
            for (int i = 0; i < Dimension - 1; i++)
            {
                sb.Append(_vector[i].ToString());
                sb.Append(",");
            }
            sb.Append(_vector[Dimension - 1].ToString());
            sb.Append("]");
            return sb.ToString();
        }
        #endregion
    }


    public class UnitNetworkVector : NetworkVector
    {
        public UnitNetworkVector(int index, int dimension)
        {
            _vector = Vector<double>.Build.Sparse(dimension);
            _vector[index] = 1.0;
        }
    }
    
    
}
