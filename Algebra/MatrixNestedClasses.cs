using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Algebra
{
    public partial class Matrix
    {
        private abstract class MatrixBase : IEquatable<MatrixBase>
        {
            #region abstract properties
            public abstract int RowCount { get; }
            public abstract int ColumnCount { get; }
            public abstract double this[int rowIndex, int columnIndex] { get; }
            #endregion

            #region abstract methods
            public abstract double[] RowArray(int index);
            public abstract double[] ColumnArray(int index);

            public abstract MatrixBase SetValueAtIndex(double value, int rowIndex, int columnIndex);
            public abstract MatrixBase Copy();
            public abstract FullMatrix AsFullMatrix();

            public abstract MatrixBase Add(MatrixBase other);
            public abstract MatrixBase Scale(double scalar);
            #endregion

            #region public methods
            public double[,] ToArray()
            {
                double[,] result = new double[RowCount, ColumnCount];
                for (int i = 0; i < RowCount; i++)
                    for (int j = 0; j < ColumnCount; j++)
                    {
                        result[i, j] = this[i, j];
                    }
                {
                }
                return result;
            }
            #endregion

            #region protected methods
            protected bool _indicesAreInRange(int row, int col)
            {
                return (row >= 0 && row < RowCount && col >= 0 && col <= ColumnCount);
            }
            #endregion


            #region IEquatable
            public override bool Equals(object other)
            {
                if (ReferenceEquals(this, other))
                    return true;

                MatrixBase omb = other as MatrixBase;

                return this.Equals(omb);
            }

            public bool Equals(MatrixBase other)
            {
                if (ReferenceEquals(this, other))
                    return true;

                if (this.RowCount != other.RowCount || this.ColumnCount != other.ColumnCount)
                    return false;

                for (int i = 0; i < RowCount; i++)
                    for (int j = 0; j < ColumnCount; j++)
                    {
                        if (this[i, j] != other[i, j])
                            return false;
                    }

                return true;
            }

            public override int GetHashCode()
            {
                int hash = 11;
                unchecked
                {
                    for (int i = 0; i < RowCount; i++)
                        for (int j = 0; j < ColumnCount; j++)
                        {
                            hash <<= 1;
                            hash ^= this[i, j].GetHashCode();
                        }
                }

                return hash;
            }
            #endregion
        }


        private class FullMatrix : MatrixBase
        {
            #region private attributes
            private double[][] _matrix;
            private int _rowCount;
            private int _columnCount;
            #endregion

            #region constructors
            public FullMatrix(IEnumerable<IEnumerable<double>> array)
            {
                if (array == null)
                    throw new ArgumentException("Cannot make matrix from null.");

                int? columnCount = array.ElementAt(0)?.Count();
                if (columnCount == null)
                    throw new ArgumentException("Cannot make matrix with a null row.");

                _matrix = new double[array.Count()][];

                int count = 0;
                foreach (IEnumerable<double> row in array)
                {
                    if (row == null)
                        throw new ArgumentException("Cannot make a matrix with a null row.");
                    if (row.Count() != columnCount)
                        throw new ArgumentException("Cannot make a matrix with different length rows.");

                    _matrix[count++] = row.ToArray();
                }
                _rowCount = count;
                _columnCount = (int)columnCount;
            }

            public FullMatrix(MatrixBase matrix)
            {
                int rowCount = matrix.RowCount;
                int columncount = matrix.ColumnCount;
                _matrix = new double[rowCount][];
                for (int i = 0; i < rowCount; i++)
                {
                    _matrix[i] = matrix.RowArray(i);
                }
            }
            #endregion


            #region VectorBase property overrides
            public override double this[int rowIndex, int columnIndex]
            {
                get { return _matrix[rowIndex][columnIndex]; }
            }

            public override int RowCount
            {
                get { return _rowCount; }
            }

            public override int ColumnCount
            {
                get { return _columnCount; }
            }
            #endregion

            #region VectorBase method overrides
            public override double[] RowArray(int index)
            {
                return _matrix[index];
            }

            public override double[] ColumnArray(int index)
            {
                double[] result = new double[RowCount];
                for (int i = 0; i < RowCount; i++)
                {
                    result[i] = _matrix[index][i];
                }
                return result;
            }

            public override MatrixBase SetValueAtIndex(double value, int rowIndex, int columnIndex)
            {
                if (_indicesAreInRange(rowIndex, columnIndex))
                {
                    _matrix[rowIndex][columnIndex] = value;
                    return this;
                }
                else
                {
                    throw new IndexOutOfRangeException();
                }
            }

            public override MatrixBase Copy()
            {
                return new FullMatrix(_matrix);
            }

            public override FullMatrix AsFullMatrix()
            {
                return this;
            }

            public override MatrixBase Add(MatrixBase other)
            {
                if (other.RowCount != this.RowCount || other.ColumnCount != this.ColumnCount)
                    throw new ArgumentException("Adding matrices of unmatched size.");

                //if (other is ZeroVector)
                //    return this;

                //if (other is BasisVector)
                //{
                //    return other.Add(this);
                //}

                int rows = this.RowCount;
                double[][] newMatrix = new double[this.RowCount][];

                for (int i = 0; i < rows; i++)
                {
                    double[] row = this.RowArray(i);
                    int rowLength = row.Length;
                    double[] newRow = new double[rowLength];
                    Array.Copy(row, newRow, rowLength);

                    for (int j = 0; j < ColumnCount; j++)
                    {
                        newRow[j] += other[i,j];
                    }

                    newMatrix[i] = newRow;
                }

                return new FullMatrix(newMatrix);
            }


            public override MatrixBase Scale(double scalar)
            {
                double[][] result = _matrix.
                    Select(
                        row => row.
                        Select(element => element * scalar).
                        ToArray()).
                    ToArray();


                return new FullMatrix(result);
            }

            #endregion
        }


    }
}
