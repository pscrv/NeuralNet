using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2
{
    public class WeightsMatrix : Matrix
    {

        #region public properties
        public int NumberOfInputs { get { return _matrix.ColumnCount; } }
        public int NumberOfOutputs { get { return _matrix.RowCount; } }
        #endregion
    }
}
