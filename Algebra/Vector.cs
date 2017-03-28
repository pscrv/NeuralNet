using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Algebra
{
    public partial class Vector : IEquatable<Vector>
    {
        #region private attributes
        private VectorBase _vector;
        #endregion

        #region public properties
        public int Length { get { return _vector.Length; } }
        #endregion


        #region constructors
        public Vector(int length)
        {
            _vector = new ZeroVector(length);
        }
        
        public Vector(double value, int index, int length)
        {
            if (index < 0 || index >= length)
                throw new IndexOutOfRangeException();

            _vector = new BasisVector(value, index, length);
        }

        public Vector(IEnumerable<double> array)
        {
            _vector = new FullVector(array.ToArray());
        }

        private Vector(VectorBase vector)
        {
            _vector = vector;
        }
        #endregion


        #region static constructors
        public static Vector MakeVector(IEnumerable<double> array)
        {
            return new Vector(new FullVector(array));
        }

        public static Vector MakeBasisVector(double value, int index, int length)
        {
            return new Vector( new BasisVector(value, index, length) );
        }

        public static Vector MakeUnitVector(int index, int length)
        {
            return new Vector(new UnitVector(index, length));
        }

        public static Vector MakeZeroVector(int length)
        {
            return new Vector(new ZeroVector(length));
        }

        public static Vector MakeCompositeVector(IEnumerable<Vector> vector)
        {
            return new Vector(
                new CompositeVector(vector.Select(x => x._vector))
                );
        }
        #endregion


        #region IEquatable
        public override bool Equals(object other)
        {
            if (ReferenceEquals(this, other))
                return true;

            Vector ov = other as Vector;
            if (ov == null)
                return false;

            return this.Equals(ov);
        }

        public bool Equals(Vector other)
        {
            if (ReferenceEquals(this, other))
                return true;

            return _vector.Equals(other._vector);
        }

        public override int GetHashCode()
        {
            return _vector.GetHashCode();
        }
        #endregion



        #region public methods
        public double this[int index]
        {
            get { return _vector[index]; }

            set { _vector.SetValueAtIndex(value, index); }
 
        }

        public double[] ToArray()
        {
            return _vector.ToArray();
        }

        public void Add(Vector other)
        {
            _vector = _vector.Add(other._vector);
        }

        public void Scale(double scalar)
        {
            _vector = _vector.Scale(scalar);
        }

        public void Subtract(Vector other)
        {
            _vector = _vector.Add(other._vector.Scale(-1.0));
        }
        #endregion




    }
}
