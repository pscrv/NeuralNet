using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet2
{
    public class VectorBatch : Matrix
    {
        #region static
        public static VectorBatch ApplyFunction(Func<double, double> fctn, VectorBatch batch)
        {
            return new VectorBatch( Matrix.ApplyFunction(fctn, batch) );
        }

        public static VectorBatch ApplyFunction(Func<double, double, double> fctn, VectorBatch first, VectorBatch second)
        {
            return new VectorBatch( Matrix.ApplyFunction(fctn, first, second) );
        }
        #endregion


        #region private attributes
        #endregion


        #region constructors
        public VectorBatch(Matrix matrix)
            : base (matrix)  { }

        public VectorBatch(Matrix<double> matrix)
            : base (matrix) { }

        //public VectorBatch(VectorBatch batch)
        //    : base (batch._matrix) { }
        #endregion


        #region public properties
        public int Dimension { get { return _matrix.ColumnCount; } }
        public int Count { get { return _matrix.RowCount; } }
        #endregion


        #region public methods
        public double this[int rowIndex, int columnIndex]
        {
            get { return _matrix[rowIndex, columnIndex]; }
        }

        public VectorBatch Add(Matrix other)
        {
            return new VectorBatch(AddMatrices(this, other));
        }

        public VectorBatch AddToEachVector(Matrix other)
        {
            return new VectorBatch(AddVectorToEachRow(this, other));
        }

        public VectorBatch Subtract(VectorBatch other)
        {
            return new VectorBatch(SubtractMatrices(this, other));
        }

        public VectorBatch Scale(double scalar)
        {
            return new VectorBatch(Scale(this, scalar));
        }

        public VectorBatch SubractVectorMaxima()
        {
            return new VectorBatch(SubtractRowMaxima(this));
        }

        public VectorBatch DivideByComponentSums()
        {
            return new VectorBatch(DivideByRowSums(this));
        }

        public Matrix SumColumnsAsMatrix()
        {
            return SumcColumns(this);
        }
        #endregion


    }
}
