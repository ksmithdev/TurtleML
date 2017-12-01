﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;

namespace TurtleML
{
    public class Tensor : IEnumerable<float>
    {
        private readonly int depth;
        private readonly int height;
        private readonly float[] values;
        private readonly int width;

        public Tensor(int width, int height, int depth)
        {
            this.width = width;
            this.height = height;
            this.depth = depth;

            values = new float[width * height * depth];
        }

        public Tensor((int width, int height, int depth) size)
            : this(size.width, size.height, size.depth)
        {
        }

        public Tensor(int size)
            : this(size, 1, 1)
        {
        }

        public Tensor(int width, int height)
            : this(width, height, 1)
        {
        }

        private Tensor(float[] values)
        {
            this.values = values;

            width = values.Length;
            height = 1;
            depth = 1;
        }

        public int Depth => depth;

        public (int, int, int) Dimensions => (width, height, depth);

        public int Height => height;

        public int Length => values.Length;

        public int Width => width;

        public float this[int i]
        {
            get { return values[i]; }
            set { values[i] = value; }
        }

        public float this[int x, int y]
        {
            get { return values[x + y * width]; }
            set { values[x + y * width] = value; }
        }

        public float this[int x, int y, int z]
        {
            get { return values[x + (y * width) + (z * width * height)]; }
            set { values[x + (y * width) + (z * width * height)] = value; }
        }

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

                Vector.Add(vector1, vector2).CopyTo(result.values);
            }

            for (; i < count; i++)
                result[i] = tensor.values[i] + array[i];

            return result;
        }

        public static Tensor Create(float[] array)
        {
            var tensor = new Tensor(array.Length);

            return tensor.Load(array);
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
                accumulator += array1[i] * array2[i];

            return accumulator;
        }

        public static Tensor Multiply(Tensor tensor, float value)
        {
            var result = new Tensor(tensor.Dimensions);

            int i = 0,
                step = Vector<float>.Count,
                count = tensor.Length;

            for (; i < count - step; i += step)
            {
                var vector = new Vector<float>(tensor.values, i);

                Vector.Multiply(vector, value).CopyTo(result.values, i);
            }

            for (; i < count; i++)
                result[i] = tensor.values[i] * value;

            return result;
        }

        public static float[] Multiply(float[] array, float value)
        {
            var result = new float[array.Length];

            int i = 0,
                step = Vector<float>.Count,
                count = array.Length;

            for (; i < count - step; i += step)
            {
                var vector = new Vector<float>(array, i);

                Vector.Multiply(vector, value).CopyTo(result, i);
            }

            for (; i < count; i++)
                result[i] = array[i] * value;

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
                values[i] += array[i];

            return this;
        }

        public void Clear()
        {
            Array.Clear(values, 0, values.Length);
        }

        public float Dot(float[] array) => Dot(values, array);

        public float Dot(Tensor tensor) => Dot(values, tensor.values);

        public IEnumerator<float> GetEnumerator()
        {
            return (values as IEnumerable<float>).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return values.GetEnumerator();
        }

        [Obsolete]
        public int IndexOf(int x, int y)
        {
            return x + y * width;
        }

        [Obsolete]
        public int IndexOf(int x, int y, int z)
        {
            return x + (y * width) + (z * width * height);
        }

        public Tensor Load(Tensor inputs)
        {
            Buffer.BlockCopy(inputs.values, 0, values, 0, values.Length * sizeof(float));

            return this;
        }

        public Tensor Load(float[] array)
        {
            Buffer.BlockCopy(array, 0, values, 0, values.Length * sizeof(float));

            return this;
        }

        public Tensor Multiply(float value)
        {
            int i = 0,
                step = Vector<float>.Count,
                count = values.Length;

            for (; i < count - step; i += step)
            {
                var vector = new Vector<float>(values, i);

                Vector.Multiply(vector, value).CopyTo(values, i);
            }

            for (; i < count; i++)
                values[i] *= value;

            return this;
        }
    }
}