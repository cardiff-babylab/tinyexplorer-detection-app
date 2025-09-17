module.exports = {
    babel: {
        plugins: [
            '@babel/plugin-transform-optional-chaining',
            '@babel/plugin-transform-nullish-coalescing-operator'
        ]
    },
    webpack: {
        configure: (webpackConfig) => {
            // Use electron-renderer target only when running in Electron context
            // For browser development, use web target
            const isElectron = process.env.ELECTRON_TARGET === 'true';
            
            if (isElectron) {
                webpackConfig.target = 'electron-renderer';
            } else {
                webpackConfig.target = 'web';
            }

            // Remove ESLint loader to avoid typescript-estree/TS version warnings during production builds
            const removeEslintFromRules = (rulesArray) => {
                if (!Array.isArray(rulesArray)) return rulesArray;
                return rulesArray
                    .map((rule) => {
                        if (!rule) return rule;

                        // CRA v3 often uses an ESLint rule with enforce: 'pre' and loader 'eslint-loader'
                        const isEslintRule = (
                            (rule.enforce === 'pre' && (rule.loader && rule.loader.includes('eslint-loader'))) ||
                            (Array.isArray(rule.use) && rule.use.some((u) => (typeof u === 'string' ? u.includes('eslint-loader') : (u && u.loader && u.loader.includes('eslint-loader'))))) ||
                            (typeof rule.loader === 'string' && rule.loader.includes('eslint-loader'))
                        );

                        if (isEslintRule) {
                            return null; // drop this rule entirely
                        }

                        // Recursively clean nested rules (e.g., oneOf)
                        if (Array.isArray(rule.oneOf)) {
                            rule.oneOf = removeEslintFromRules(rule.oneOf).filter(Boolean);
                        }

                        // Clean rule.use arrays of any eslint-loader entries
                        if (Array.isArray(rule.use)) {
                            rule.use = rule.use.filter((u) => {
                                if (!u) return false;
                                if (typeof u === 'string') {
                                    return !u.includes('eslint-loader');
                                }
                                return !(u.loader && u.loader.includes('eslint-loader'));
                            });
                        }

                        return rule;
                    })
                    .filter(Boolean);
            };

            if (webpackConfig && webpackConfig.module && Array.isArray(webpackConfig.module.rules)) {
                webpackConfig.module.rules = removeEslintFromRules(webpackConfig.module.rules);
            }
            
            return webpackConfig;
        }
    }
};
