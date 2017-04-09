using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet2
{
    public class DataVector : VectorBatch
    {
        #region constructors
        public DataVector(Vector<double> vector)
            : base (Matrix<double>.Build.DenseOfRowVectors(vector))
        { }

        public DataVector(int dimension)
            : base (Matrix<double>.Build.Dense(1, dimension))
        { }

        public DataVector(VectorBatch batch)
            : base (batch)
        {
            if (batch.Count != 1)
                throw new ArgumentException("Can create a Vector only from a non-empty, singleton VectorBatch");
        }

        protected DataVector(Matrix matrix)
            : base (matrix) { }
        #endregion

        #region public methods
        public double this[int index]
        {
            get { return _matrix[0, index]; }
        }

        public DataVector Add(DataVector other)
        {
            return new DataVector(AddMatrices(this, other).Row(0));
        }
        
        public DataVector Subtract(DataVector other)
        {
            return new DataVector(SubtractMatrices(this, other).Row(0));
        }
        #endregion
        
    }
}
