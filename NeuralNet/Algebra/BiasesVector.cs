using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;


namespace NeuralNet2
{
    public class BiasesVector
    {
        #region attributes
        private Vector<double> _vector;
        #endregion

        #region public properties
        public int Dimension { get { return _vector.Count; } }
        #endregion
    }
}
