import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import dts from 'rollup-plugin-dts';
import postcss from 'rollup-plugin-postcss';
import { readFileSync } from 'fs';

const packageJson = JSON.parse(readFileSync('./package.json', 'utf-8'));

export default [
  // Main build configuration
  {
    input: 'src/index.ts',
    output: [
      {
        file: packageJson.main,
        format: 'cjs',
        sourcemap: true,
        exports: 'named'
      },
      {
        file: packageJson.module,
        format: 'esm',
        sourcemap: true,
        exports: 'named'
      }
    ],
    plugins: [
      resolve({
        browser: true,
        preferBuiltins: false
      }),
      commonjs(),
      postcss({
        extract: 'styles.css',
        minimize: true,
        use: ['sass']
      }),
      typescript({
        tsconfig: './tsconfig.json',
        declaration: false,
        declarationMap: false
      })
    ],
    external: [
      'react',
      'react-dom',
      'react/jsx-runtime',
      ...Object.keys(packageJson.peerDependencies || {})
    ]
  },

  // Type definitions
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/index.d.ts',
      format: 'esm'
    },
    plugins: [dts()],
    external: [/\.css$/]
  },
];
