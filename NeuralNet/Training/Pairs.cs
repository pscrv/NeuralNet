using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class VectorPair
    {
        #region protected attributes
        public NetworkVector First { get; protected set; }
        public NetworkVector Second { get; protected set; }
        #endregion

        #region constructors
        public VectorPair(NetworkVector a, NetworkVector b)
        {
            First = a;
            Second = b;
        }
        #endregion
    }
    
    public class BatchPair
    {
        #region prottected attributes
        public VectorBatch First { get; protected set; }
        public VectorBatch Second { get; protected set; }
        #endregion

        #region constructors
        public BatchPair(VectorBatch a, VectorBatch b)
        {
            First = a;
            Second = b;
        }
        #endregion
    }
}
