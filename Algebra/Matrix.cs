using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Algebra
{
    public partial class Matrix : IEquatable<Matrix>
    {
        #region private attributes
        private MatrixBase _matrix;
        #endregion

        #region public properties
        public int RowCount { get { return _matrix.RowCount; } }
        public int ColumnCount { get { return _matrix.ColumnCount; } }
        #endregion


        #region constructors

        public Matrix(IEnumerable<IEnumerable<double>> array)
        {
            _matrix = new FullMatrix(array);
        }

        private Matrix(MatrixBase matrix)
        {
            _matrix = matrix;
        }
        #endregion


        #region static constructors
        //public static Vector MakeVector(IEnumerable<double> array)
        //{
        //    return new Vector(new FullVector(array));
        //}

        //public static Vector MakeBasisVector(double value, int index, int length)
        //{
        //    return new Vector(new BasisVector(value, index, length));
        //}

        //public static Vector MakeUnitVector(int index, int length)
        //{
        //    return new Vector(new UnitVector(index, length));
        //}

        //public static Vector MakeZeroVector(int length)
        //{
        //    return new Vector(new ZeroVector(length));
        //}

        //public static Vector MakeCompositeVector(IEnumerable<Vector> vector)
        //{
        //    return new Vector(
        //        new CompositeVector(vector.Select(x => x._vector))
        //        );
        //}
        #endregion


        #region IEquatable
        public override bool Equals(object other)
        {
            if (ReferenceEquals(this, other))
                return true;

            Matrix om = other as Matrix;
            if (om == null)
                return false;

            return this.Equals(om);
        }

        public bool Equals(Matrix other)
        {
            if (ReferenceEquals(this, other))
                return true;

            return _matrix.Equals(other._matrix);
        }

        public override int GetHashCode()
        {
            return _matrix.GetHashCode();
        }
        #endregion



        #region public methods
        public double this[int rowIndex, int columnIndex]
        {
            get { return _matrix[rowIndex, columnIndex]; }

            set { _matrix.SetValueAtIndex(value, rowIndex, columnIndex); }

        }

        public double[,] ToArray()
        {
            return _matrix.ToArray();
        }

        public void Add(Matrix other)
        {
            _matrix = _matrix.Add(other._matrix);
        }

        public void Scale(double scalar)
        {
            _matrix = _matrix.Scale(scalar);
        }

        public void Subtract(Matrix other)
        {
            _matrix = _matrix.Add(other._matrix.Scale(-1.0));
        }
        #endregion




    }
}
