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
        #region private attributes
        #endregion

        #region constructors
        protected VectorBatch() { }

        public VectorBatch(Matrix<double> matrix)
        {
            matrix.CopyTo(_matrix);
        }

        public VectorBatch(VectorBatch batch)
        {
            batch._matrix.CopyTo(this._matrix);
        }
        #endregion


        #region public properties
        public int Count { get { return _matrix.RowCount; } }
        #endregion



        #region public methods

        #endregion


    }
}
