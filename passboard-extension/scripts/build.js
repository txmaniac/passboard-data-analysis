const esbuild = require('esbuild');
const { exec } = require('child_process');
const path = require('path');

async function build() {
    console.log('Building content script...');
    try {
        await esbuild.build({
            entryPoints: ['scripts/content.jsx'],
            bundle: true,
            outfile: 'public/content.js',
            format: 'iife',
            loader: { '.js': 'jsx', '.jsx': 'jsx' },
            minify: false, // keep readable for debug
            sourcemap: true,
            define: {
                'process.env.NODE_ENV': '"development"'
            },
            // Ensure React is bundled
            inject: []
        });
        console.log('Content script built successfully.');
    } catch (e) {
        console.error('Content script build failed:', e);
        process.exit(1);
    }

    console.log('Building Tailwind CSS...');
    exec('npx @tailwindcss/cli -i ./app/globals.css -o ./public/extension.css', (err, stdout, stderr) => {
        if (err) {
            console.error('Tailwind build failed:', stderr);
            return;
        }
        console.log('Tailwind CSS built successfully.');
    });
}

build();
