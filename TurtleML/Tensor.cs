using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace TurtleML
{
    public class Tensor : IEnumerable<float>
    {
        private readonly float[] values;

        public Tensor(int width, int height, int depth)
        {
            Width = width;
            Height = height;
            Depth = depth;

            values = new float[width * height * depth];
        }

        public Tensor((int width, int height, int depth) size)
            : this(size.width, size.height, size.depth)
        {
        }

        public Tensor(int length)
            : this(length, 1, 1)
        {
        }

        public Tensor(int width, int height)
            : this(width, height, 1)
        {
        }

        private Tensor(float[] values)
        {
            this.values = values;

            Width = values.Length;
            Height = 1;
            Depth = 1;
        }

        public int Depth { get; private set; }

        public (int, int, int) Dimensions => (Width, Height, Depth);

        public int Height { get; private set; }

        public int Length => values.Length;

        public int Width { get; private set; }

        public ref float this[int i] => ref values[i];

        public ref float this[int x, int y] => ref this[IndexOf(x, y)];

        public ref float this[int x, int y, int z] => ref this[IndexOf(x, y, z)];

        public static Tensor Add(Tensor tensor1, Tensor tensor2) => Add(tensor1, tensor2.values);

        public static Tensor Add(Tensor tensor, float[] array)
        {
            var result = new Tensor(tensor.Dimensions);

            int i = 0,
                step = Vector<float>.Count,
                count = tensor.Length;

            for (; i < count - step; i += step)
            {
                var vector1 = new Vector<float>(tensor.values, i);
                var vector2 = new Vector<float>(array, i);

                Vector.Add(vector1, vector2).CopyTo(result.values, i);
            }

            for (; i < count; i++)
            {
                result[i] = tensor.values[i] + array[i];
            }

            return result;
        }

        public static void Copy(Tensor src, int srcX, int srcY, int srcZ, Tensor dst, int dstX, int dstY, int dstZ, int count)
        {
            var srcOffset = src.IndexOf(srcX, srcY, srcZ);
            var dstOffset = dst.IndexOf(dstX, dstY, dstZ);

            Array.Copy(src.values, srcOffset, dst.values, dstOffset, count);
        }

        public static Tensor Create(float[] array)
        {
            var tensor = new Tensor(array.Length);
            tensor.Load(array);
            return tensor;
        }

        public static Tensor Create(Tensor tensor)
        {
            return Create(tensor.values);
        }

        public static float Dot(Tensor tensor1, Tensor tensor2) => Dot(tensor1.values, tensor2.values);

        public static float Dot(float[] array1, float[] array2)
        {
            float accumulator = 0f;

            int i = 0,
                step = Vector<float>.Count,
                count = Math.Min(array1.Length, array2.Length);

            for (; i < count - step; i += step)
            {
                var vector1 = new Vector<float>(array1, i);
                var vector2 = new Vector<float>(array2, i);

                accumulator += Vector.Dot(vector1, vector2);
            }

            for (; i < count; i++)
            {
                accumulator += array1[i] * array2[i];
            }

            return accumulator;
        }

        public static Tensor Multiply(Tensor tensor, float value)
        {
            var result = new Tensor(tensor.Dimensions);
            Multiply(tensor, value, result);
            return result;
        }

        public static void Multiply(Tensor tensor, float value, Tensor result)
        {
            int i = 0,
                step = Vector<float>.Count,
                count = tensor.Length;

            for (; i < count - step; i += step)
            {
                var vector = new Vector<float>(tensor.values, i);

                Vector.Multiply(vector, value).CopyTo(result.values, i);
            }

            for (; i < count; i++)
            {
                result[i] = tensor.values[i] * value;
            }
        }

        public static void Multiply(Tensor tensor, float[] array, Tensor result)
        {
            int i = 0,
                step = Vector<float>.Count,
                count = tensor.Length;

            for (; i < count - step; i += step)
            {
                var vector1 = new Vector<float>(tensor.values, i);
                var vector2 = new Vector<float>(array, i);

                Vector.Multiply(vector1, vector2).CopyTo(result.values, i);
            }

            for (; i < count; i++)
            {
                result[i] = tensor.values[i] * array[i];
            }
        }

        public static Tensor Multiply(Tensor tensor1, Tensor tensor2) => Multiply(tensor1, tensor2.values);

        public static Tensor Multiply(Tensor tensor, float[] array)
        {
            var result = new Tensor(tensor.Dimensions);
            Multiply(tensor, array, result);
            return result;
        }

        public static Tensor Wrap(float[] values)
        {
            return new Tensor(values);
        }

        public Tensor Add(Tensor tensor) => Add(tensor.values);

        public Tensor Add(float[] array)
        {
            int i = 0,
                step = Vector<float>.Count,
                count = values.Length;

            for (; i < count - step; i += step)
            {
                var vector1 = new Vector<float>(values, i);
                var vector2 = new Vector<float>(array, i);

                Vector.Add(vector1, vector2).CopyTo(values, i);
            }

            for (; i < count; i++)
            {
                values[i] += array[i];
            }

            return this;
        }

        public void Clear(float value)
        {
            for (int i = 0, count = values.Length; i < count; i++)
            {
                values[i] = value;
            }
        }

        public void Clear()
        {
            Array.Clear(values, 0, values.Length);
        }

        public void CopyTo(Tensor tensor, int offset) => CopyTo(tensor.values, offset);

        public void CopyTo(float[] array, int offset)
        {
            Array.Copy(values, 0, array, offset, values.Length);
        }

        public float Dot(float[] array) => Dot(values, array);

        public float Dot(Tensor tensor) => Dot(values, tensor.values);

        public IEnumerator<float> GetEnumerator()
        {
            return ((IEnumerable<float>)values).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return values.GetEnumerator();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int IndexOf(int x, int y)
        {
            return x + (y * Width);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int IndexOf(int x, int y, int z)
        {
            return x + (y * Width) + (z * Width * Height);
        }

        public void Load(Tensor source, int sourceOffset) => Load(source.values, sourceOffset);

        public void Load(float[] source, int sourceOffset)
        {
            Buffer.BlockCopy(source, sourceOffset, values, 0, values.Length * sizeof(float));
        }

        public void Load(Tensor source) => Load(source.values);

        public void Load(float[] source)
        {
            Buffer.BlockCopy(source, 0, values, 0, values.Length * sizeof(float));
        }

        public Tensor Multiply(float value)
        {
            Multiply(this, value, this);

            return this;
        }

        public Tensor Multiply(Tensor tensor) => Multiply(tensor.values);

        public Tensor Multiply(float[] array)
        {
            Multiply(this, array, this);

            return this;
        }

        public Tensor Reshape(int width, int height, int depth)
        {
            Width = width;
            Height = height;
            Depth = depth;

            return this;
        }

#if NETSTANDARD2_1_OR_GREATER
        public static float Dot(Span<float> span1, Span<float> span2)
        {
            var vector1 = new Vector<float>(span1);
            var vector2 = new Vector<float>(span2);

            return Vector.Dot(vector1, vector2);
        }
#endif
    }
}