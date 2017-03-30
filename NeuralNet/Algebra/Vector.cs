using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class Vector : IEquatable<Vector>
    {     
        #region abstract properties
        public abstract int Length { get; }
        public abstract double this[int index] { get; set; }
        #endregion

        #region abstract methods
        public abstract double[] ToArray();
        public abstract Vector SumWith(Vector vectorToAdd);
        public abstract Vector ScaleBy(double scalar);
        public abstract Vector LeftMultiplyBy(WeightsMatrix matrix);
        public abstract FullVector AsFullVector();
        #endregion

        

        #region IEquatable
        public override bool Equals(object other)
        {
            if (ReferenceEquals(null, other))
                return false;

            if (ReferenceEquals(other, this))
                return true;

            if (!(other is Vector))
                return false;

            return this.Equals(other as Vector);
        }

        public bool Equals(Vector other)
        {
            if (other == null)
                return false;

            if (this.Length != other.Length)
                return false;

            double epsilon = 0.000000001;
            for (int i = 0; i < this.Length; i++)
            {
                double difference = Math.Abs(this[i] - other[i]);
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
                for (int i = 0; i < Length; i++)
                {
                    hash <<= 1;
                    hash ^= this[i].GetHashCode();
                }
            }
            return hash;
        }
        #endregion     
    }



    public class FullVector : Vector
    {
        #region private attributes
        private double[] _vector;
        #endregion

        #region constructors
        public FullVector(double[] vector)
        {
            _vector = vector;
        }

        public FullVector(int length)
        {
            _vector = new double[length];
        }
                
        public FullVector(Vector vector)
        {
            int length = vector.Length;
            _vector = new double[length];
            for (int i = 0; i < length; i++)
            {
                _vector[i] = vector[i];
            }
        }
        #endregion


        #region Vector property overrides
        public override double this[int index]
        {
            get { return _vector[index]; }
            set { _vector[index] = value; }
        }

        public override int Length
        {
            get { return _vector.Length; }
        }
        #endregion

        #region Vector method overrides
        public override double[] ToArray()
        {
            return (double[]) _vector.Clone();
        }

        public override Vector SumWith(Vector vectorToAdd)
        {
            Vector result = new FullVector(this);
            for (int i = 0; i < Length; i++)
            {
                result[i] += vectorToAdd[i];
            }
            return result;
        }

        public override Vector ScaleBy(double scalar)
        {
            Vector result = new FullVector(this);
            if (scalar == 1.0)
                return result;

            for (int i = 0; i < Length; i++)
            {
                result[i] *= scalar;
            }
            return result;
        }

        public override Vector LeftMultiplyBy(WeightsMatrix matrix)
        {
            Vector result = new FullVector(matrix.NumberOfOutputs);
            double[,] matrixArray = matrix.ToArray();
            for (int i = 0; i < matrix.NumberOfInputs; i++)
            {
                for (int j = 0; j < matrix.NumberOfOutputs; j++)
                {
                    result[j] += matrixArray[j, i] * this[i];
                }
            }

            return result;

            // Look up https://msdn.microsoft.com/en-us/library/ff963547.aspx
            // to see how to parallelize this
        }

        public override FullVector AsFullVector()
        {
            return this;
        }
        #endregion

    }


    public class UnitVector : Vector
    {
        #region private attributes
        private int _length;
        private int _index;
        #endregion

        #region public properties
        public int Index { get { return _index; } }        
        #endregion


        #region constructors
        public UnitVector(int index, int length)
        {
            if (length < 1)
                throw new ArgumentNullException("Attemp to create a UnitVector of length < 1.");

            if (index < 0 || index >= length)
                throw new IndexOutOfRangeException();

            _length = length;
            _index = index;
        }
        #endregion

        #region Vector property overrides
        public override double this[int index]
        {
            get
            {
                if (index < 0 || index >= Length)
                    throw new ArgumentOutOfRangeException();
                return (index == _index) ? 1.0 : 0.0;
            }

            set { throw new InvalidOperationException("Cannot write to a UnitVector."); }
        }

        public override int Length
        {
            get { return _length; }
        }
        #endregion

        #region Vector method overrides
        public override double[] ToArray()
        {
            double[] result = new double[_length];
            result[_index] = 1.0;
            return result;
        }

        public override Vector SumWith(Vector vectorToAdd)
        {
            Vector result = new FullVector(vectorToAdd);
            result[_index] += 1.0;
            return result;
        }

        public override Vector ScaleBy(double scalar)
        {
            return new FullVector(this).ScaleBy(scalar);
        }

        public override Vector LeftMultiplyBy(WeightsMatrix matrix)
        {
            Vector result = new FullVector(matrix.NumberOfOutputs);
            double[,] matrixArray = matrix.ToArray();
            for (int i = 0; i < matrix.NumberOfOutputs; i++)
            {
                result[i] = matrixArray[i, _index];
            }
            return result;
        }

        public override FullVector AsFullVector()
        {
            return new FullVector(this);
        }
        #endregion
    }


    public class CompositeVector : Vector
    {
        List<Vector> _vectors;

        public CompositeVector(IEnumerable<Vector> vectors)
        {
            _vectors = vectors.ToList();
        }


        public override double this[int index]
        {
            get
            {
                if (index < 0)
                    throw new ArgumentOutOfRangeException();

                int count = 0;
                foreach (Vector vector in _vectors)
                {
                    if (count + vector.Length > index)
                        return vector[index - count];
                    count++;
                }
                throw new ArgumentOutOfRangeException();
            }

            set
            {
                throw new InvalidOperationException("Cannot set a unit vector, it is read-only.");
            }
        }

        public override int Length
        {
            get { return _vectors.Sum(x => x.Length); }
        }

        public override Vector LeftMultiplyBy(WeightsMatrix matrix)
        {
            return AsFullVector().LeftMultiplyBy(matrix);
        }

        public override Vector ScaleBy(double scalar)
        {
            return AsFullVector().ScaleBy(scalar);
        }

        public override Vector SumWith(Vector vectorToAdd)
        {
            return AsFullVector().SumWith(vectorToAdd);
        }

        public override double[] ToArray()
        {
            return AsFullVector().ToArray();
        }

        public override FullVector AsFullVector()
        {
            FullVector result = new FullVector(Length);
            int index = 0;
            foreach (Vector vector in _vectors)
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    result[index + i] = vector[i];
                }

                index += vector.Length;
            }
            return result;
        }
    }
}
