using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet2
{
    public class WeightsMatrix : Matrix
    {
        #region static
        public static WeightsMatrix FromVectorBatchPair(VectorBatch first, VectorBatch second)
        {
            return new WeightsMatrix(Matrix.TransposeAndMultiplyBy(second, first));
        }
        #endregion


        #region constructors
        public WeightsMatrix(Matrix<double> matrix)
            : base (matrix) { }

        public WeightsMatrix(Matrix matrix)
            : base (matrix) { }
        #endregion


        #region public properties
        public int NumberOfInputs { get { return _matrix.ColumnCount; } }
        public int NumberOfOutputs { get { return _matrix.RowCount; } }
        #endregion


        #region public methods
        public WeightsMatrix Add(WeightsMatrix other)
        {
            return new WeightsMatrix(AddMatrices(this, other));
        }

        public WeightsMatrix Scale(double scalar)
        {
            return new WeightsMatrix(Scale(this, scalar));
        }

        public VectorBatch ApplyForwards(VectorBatch vector)
        {
            return new VectorBatch( (Matrix.MultiplyByTranspose(vector, this)) );
        }

        public VectorBatch ApplyBackwards(VectorBatch vector)
        {
            return new VectorBatch((Matrix.Multiply(vector, this)));
        }
        #endregion
    }
}
