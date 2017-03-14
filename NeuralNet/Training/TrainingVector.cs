using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class VectorPair
    {
        #region protected members
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


    public class TrainingVector : VectorPair
    {

        #region constructors
        public TrainingVector(NetworkVector input, NetworkVector target)
            : base (input, target)
        { }
        #endregion

        #region public properties
        public NetworkVector Input { get { return First; } }
        public NetworkVector Target { get { return Second; } }
        #endregion
    }
}
