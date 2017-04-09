using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;


namespace NeuralNet2
{
    public class BiasesVector : Matrix
    {
        #region attributes
        #endregion


        #region constructors
        public BiasesVector(int dimension)
            : base(1, dimension) { }

        public BiasesVector(Vector<double> vector)
            : base (Matrix<double>.Build.DenseOfRowVectors(vector)) { }

        public BiasesVector(Matrix matrix)
            : base (matrix) { }
        #endregion


        #region public properties
        public double this[int index]
        {
            get { return _matrix[0, index]; }
        }

        public int Dimension { get { return _matrix.ColumnCount; } }
        #endregion


        #region public methods
        public BiasesVector Add(Matrix other)
        {
            return new BiasesVector(AddMatrices(this, other).Row(0));
        }

        public BiasesVector Subtract(BiasesVector other)
        {
            return new BiasesVector(SubtractMatrices(this, other).Row(0));
        }

        public BiasesVector Scale(double scalar)
        {
            return new BiasesVector(Scale(this, scalar));
        }
        #endregion
    }
}
