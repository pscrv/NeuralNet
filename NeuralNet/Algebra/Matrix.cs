using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet2
{
    public class Matrix : IEquatable<Matrix>
    {
        #region static
        protected static Matrix ApplyFunction(Func<double, double> fctn, Matrix matrix)
        {
            return new Matrix( matrix._matrix.Map(fctn) );
        }

        protected static Matrix ApplyFunction(Func<double, double, double> fctn, Matrix matrix1, Matrix matrix2)
        {
            return new Matrix(matrix1._matrix.Map2(fctn, matrix2._matrix));
        }

        protected static Matrix Scale(Matrix matrix, double scalar)
        {
            return new Matrix(matrix._matrix.Multiply(scalar));
        }

        protected static Matrix Multiply(Matrix first, Matrix second)
        {
            return new Matrix(first._matrix.Multiply(second._matrix));
        }

        protected static Matrix MultiplyByTranspose(Matrix first, Matrix second)
        {
            return new Matrix(first._matrix.Multiply(second._matrix.Transpose()));
        }

        protected static Matrix TransposeAndMultiplyBy(Matrix first, Matrix second)
        {
            return new Matrix(first._matrix.Transpose().Multiply(second._matrix));
        }

        protected static Matrix AddVectorToEachRow(Matrix matrix, Matrix vector)
        {
            Matrix result = new Matrix(matrix._matrix.RowCount, matrix._matrix.ColumnCount);
            for (int i = 0; i < result._matrix.RowCount; i++)
            {
                result._matrix.SetRow(i, matrix._matrix.Row(i).Add(vector._matrix.Row(0)));
            }
            return result;
        }

        protected static Matrix SubtractRowMaxima(Matrix matrix)
        {
            Matrix result = new Matrix(matrix._matrix.RowCount, matrix._matrix.ColumnCount);
            int rowIndex = 0;
            foreach (Vector<double> row in matrix._matrix.EnumerateRows())
            {
                result._matrix.SetRow(rowIndex, row.Subtract(row.Max()) );
                rowIndex++;
            }

            return result;
        }

        protected static Matrix DivideByRowSums(Matrix matrix)
        {
            Matrix result = new Matrix(matrix._matrix.RowCount, matrix._matrix.ColumnCount);
            int rowIndex = 0;
            foreach (Vector<double> row in matrix._matrix.EnumerateRows())
            {
                result._matrix.SetRow(rowIndex, row.Divide(row.Sum()));
                rowIndex++;
            }

            return result;
        }

        protected static Matrix SumcColumns(Matrix matrix)
        {
            return new Matrix ( matrix._matrix.ColumnSums() );
        }




        protected static Matrix<double> AddMatrices(Matrix first, Matrix second)
        {
            return first._matrix.Add(second._matrix);
        }

        protected static Matrix<double> SubtractMatrices(Matrix first, Matrix second)
        {
            return first._matrix.Subtract(second._matrix);
        }
        #endregion


        #region attributes
        protected Matrix<double> _matrix;
        #endregion


        #region constructors
        protected Matrix(int numberOfRows, int numberOfColumns)
        {
            _matrix = Matrix<double>.Build.Dense(numberOfRows, numberOfColumns);
        }

        protected Matrix(Matrix<double> matrix)
            : this (matrix.RowCount, matrix.ColumnCount)
        {
            matrix.CopyTo(_matrix);
        }

        protected Matrix(Matrix matrix)
            : this(matrix._matrix) { }

        protected Matrix(Vector<double> vector)
            : this (1, vector.Count)
        {
            _matrix.SetRow(0, vector);
        }
        #endregion


        #region public methods
        public double Max()
        {
            return _matrix.Enumerate().Max();
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

            return this.Equals(other as Matrix);
        }

        public bool Equals(Matrix other)
        {
            if (other == null)
                return false;

            return this._matrix.Equals(other._matrix);
        }

        public override int GetHashCode()
        {
            return _matrix.GetHashCode();
        }
        #endregion

    }
}
