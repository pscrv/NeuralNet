using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Algebra
{
    public partial class Vector
    {
        #region private classes
        private abstract class VectorBase : IEquatable<VectorBase>
        {
            #region abstract properties
            public abstract int Length { get; }
            public abstract double this[int index] { get; }
            #endregion

            #region abstract methods
            public abstract VectorBase SetValueAtIndex(double value, int index);
            public abstract VectorBase Copy();
            public abstract FullVector AsFullVector();

            public abstract VectorBase Add(VectorBase other);
            public abstract VectorBase Scale(double scalar);
            #endregion

            #region public methods
            public double[] ToArray()
            {
                double[] result = new double[Length];
                for (int i = 0; i < Length; i++)
                {
                    result[i] = this[i];
                }
                return result;
            }
            #endregion

            #region protected methods
            protected bool _indexIsInRange(int index)
            {
                return (index >= 0 && index < Length);
            }
            #endregion


            #region IEquatable
            public override bool Equals(object other)
            {
                if (ReferenceEquals(this, other))
                    return true;

                VectorBase ov = other as VectorBase;

                return this.Equals(ov);
            }

            public bool Equals(VectorBase other)
            {
                if (ReferenceEquals(this, other))
                    return true;

                if (this.Length != other.Length)
                    return false;

                for (int i = 0; i < Length; i++)
                {
                    if (this[i] != other[i])
                        return false;
                }

                return true;
            }

            public override int GetHashCode()
            {
                int hash = 11;
                unchecked
                {
                    for (int i = 0; i < Length; i++)
                    {
                        hash <<= 1;
                        hash ^= this[i].GetHashCode();
                    }
                }
                return hash;
            }
            #endregion
        }


        private class FullVector : VectorBase
        {
            #region private attributes
            private double[] _vector;
            #endregion

            #region constructors
            public FullVector(IEnumerable<double> array)
            {
                _vector = array.ToArray();
            }

            public FullVector(int length)
                : this(new double[length])
            { }

            public FullVector(VectorBase vector)
            {
                int length = vector.Length;
                _vector = new double[length];
                for (int i = 0; i < length; i++)
                {
                    _vector[i] = vector[i];
                }
            }
            #endregion


            #region VectorBase property overrides
            public override double this[int index]
            {
                get { return _vector[index]; }
            }

            public override int Length
            {
                get { return _vector.Length; }
            }
            #endregion

            #region VectorBase method overrides
            public override VectorBase SetValueAtIndex(double value, int index)
            {
                if (_indexIsInRange(index))
                {
                    _vector[index] = value;
                    return this;
                }
                else
                {
                    throw new IndexOutOfRangeException();
                }
            }

            public override VectorBase Copy()
            {
                return new FullVector(_vector);
            }

            public override FullVector AsFullVector()
            {
                return this;
            }

            public override VectorBase Add(VectorBase other)
            {
                if (other.Length != this.Length)
                    throw new ArgumentException("Adding vectors of unequal length.");

                if (other is ZeroVector)
                    return this;

                if (other is BasisVector)
                {
                    return other.Add(this);
                }


                double[] result = new double[Length];

                for (int i = 0; i < Length; i++)
                {
                    result[i] = this[i] + other[i];
                }
                return new FullVector(result);
            }

            public override VectorBase Scale(double scalar)
            {
                double[] result = _vector.Select(x => x * scalar).ToArray();
                return new FullVector(result);
            }

            #endregion
        }


        private class BasisVector : VectorBase
        {
            #region private attributes
            protected int _length;
            protected int _index;
            protected double _value;
            #endregion

            #region public properties
            public int Index { get { return _index; } }
            public double Value { get { return _value; } }
            #endregion


            #region constructors
            public BasisVector(double value, int index, int length)
            {
                if (length < 1)
                    throw new ArgumentNullException("Attemp to create a UnitVector of length < 1.");

                if (index < 0 || index >= length)
                    throw new IndexOutOfRangeException();

                _length = length;
                _index = index;
                _value = value;
            }
            #endregion

            #region Vector property overrides
            public override double this[int index]
            {
                get
                {
                    if (index < 0 || index >= Length)
                        throw new IndexOutOfRangeException();
                    return (index == _index) ? _value : 0.0;
                }
            }

            public override int Length
            {
                get { return _length; }
            }
            #endregion

            #region Vector method overrides
            public override VectorBase SetValueAtIndex(double value, int index)
            {
                if (index == _index)
                {
                    _value = value;
                    return this;
                }
                else if (_indexIsInRange(index))
                {
                    VectorBase result = this.AsFullVector();
                    result.SetValueAtIndex(value, index);
                    return result;
                }
                else
                {
                    throw new IndexOutOfRangeException();
                }
            }

            public override VectorBase Copy()
            {
                return new BasisVector(_value, _index, _length);
            }

            public override FullVector AsFullVector()
            {
                return new FullVector(this);
            }

            public override VectorBase Add(VectorBase other)
            {
                if (other.Length != this.Length)
                    throw new ArgumentException("Unequal vector lengths.");

                if (other is ZeroVector)
                    return this;

                if (other is BasisVector)
                {
                    BasisVector obv = other as BasisVector;
                    if (obv.Index == this.Index)
                        return new BasisVector(obv.Value + this.Value, Index, Length);

                }

                VectorBase result = other.Copy();
                result.SetValueAtIndex(result[Index] + Value, Index);
                return result;

            }

            public override VectorBase Scale(double scalar)
            {
                return new BasisVector(_value * scalar, Index, Length);
            }
            #endregion
        }


        private class UnitVector : BasisVector
        {
            public UnitVector(int index, int length)
                : base(1.0, index, length) { }
        }


        private class ZeroVector : VectorBase
        {
            private int _length;

            public ZeroVector(int length)
            {
                _length = length;
            }


            #region Vector property overrides
            public override double this[int index]
            {
                get
                {
                    if (index < 0 || index >= Length)
                        throw new IndexOutOfRangeException();
                    return 0.0;
                }
            }

            public override int Length
            {
                get
                {
                    return _length;
                }
            }
            #endregion

            #region Vector method overrides
            public override VectorBase SetValueAtIndex(double value, int index)
            {
                if (_indexIsInRange(index))
                {
                    return new BasisVector(value, index, Length);
                }
                else
                {
                    throw new IndexOutOfRangeException();
                }
            }

            public override VectorBase Copy()
            {
                return new ZeroVector(_length);
            }

            public override FullVector AsFullVector()
            {
                return new FullVector(this);
            }

            public override VectorBase Add(VectorBase other)
            {
                if (other.Length != this.Length)
                    throw new ArgumentException("Unequal vector lengths.");

                return other;
            }

            public override VectorBase Scale(double scalar)
            {
                return new ZeroVector(Length);
            }
            #endregion
        }


        private class CompositeVector : VectorBase
        {
            #region private attributes
            List<VectorBase> _vectors;
            #endregion

            #region constructors
            public CompositeVector(IEnumerable<VectorBase> vectors)
            {
                _vectors = vectors.ToList();
            }
            #endregion

            #region Vector property overrides
            public override double this[int index]
            {
                get
                {
                    var x = _skipToVectorAndIndex(index);
                    return x.Item1[x.Item2];
                }
            }

            public override int Length
            {
                get { return _vectors.Sum(x => x.Length); }
            }
            #endregion

            #region Vector method overrides
            public override VectorBase SetValueAtIndex(double value, int index)
            {
                var x = _skipToVectorAndIndex(index);
                x.Item1.SetValueAtIndex(value, x.Item2);
                return this;
            }

            public override VectorBase Copy()
            {
                return new CompositeVector(_vectors.Select(x => x.Copy()));
            }

            public override VectorBase Add(VectorBase other)
            {
                if (other.Length != this.Length)
                    throw new ArgumentException("Unequal vector lengths");

                CompositeVector ocv = other as CompositeVector;
                if (ocv != null && ocv._componentLengths() == this._componentLengths())
                {
                    return new CompositeVector(
                        _vectors.Zip(
                            ocv._vectors,
                            (VectorBase a, VectorBase b) => a.Add(b)
                            )
                        );
                }

                double[] result = new double[Length];
                for (int i = 0; i < Length; i++)
                {
                    result[i] = this[i] + other[i];
                }
                return new FullVector(result);
            }

            public override VectorBase Scale(double scalar)
            {
                return new CompositeVector(_vectors.Select(x => x.Scale(scalar)).ToList());
            }

            public override FullVector AsFullVector()
            {
                FullVector result = new FullVector(Length);
                int index = 0;
                foreach (VectorBase vector in _vectors)
                {
                    for (int i = 0; i < vector.Length; i++)
                    {
                        result.SetValueAtIndex(vector[i], index + i);
                    }

                    index += vector.Length;
                }
                return result;
            }
            #endregion


            #region private methods
            private Tuple<VectorBase, int> _skipToVectorAndIndex(int index)
            {
                if (index < 0)
                    throw new IndexOutOfRangeException();

                int count = 0;
                foreach (VectorBase vector in _vectors)
                {
                    if (count + vector.Length > index)
                    {
                        return new Tuple<VectorBase, int>(vector, index - count);
                    }
                    count += vector.Length;
                }
                throw new IndexOutOfRangeException();
            }

            private List<int> _componentLengths()
            {
                return _vectors.Select(x => x.Length).ToList();
            }
            #endregion
        }
        #endregion

    }
}
