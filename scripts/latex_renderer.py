import os
import re
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib for math rendering
rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font (TeX-like)
rcParams['font.family'] = 'serif'

def convert_latex_to_matplotlib(latex_str):
    """
    Convert LaTeX commands to Matplotlib-compatible math syntax.
    
    Args:
        latex_str: LaTeX string to convert
        
    Returns:
        str: Matplotlib-compatible math string
    """
    # Replace LaTeX commands that aren't directly supported by Matplotlib
    replacements = {
        r'\\ge': r'\\geq',     # \ge -> \geq
        r'\\le': r'\\leq',     # \le -> \leq
        r'\\to': r'\\rightarrow',  # \to -> \rightarrow
        r'\\ldots': r'\\dots',  # \ldots -> \dots
        r'\\iff': r'\\Leftrightarrow',  # \iff -> \Leftrightarrow
        r'\\mod': r'\\bmod',    # \mod -> \bmod
        r'\\colon': r':',      # \colon -> :
        r'\\textbf\{([^}]*)\}': r'\\mathbf{\1}',  # \textbf{} -> \mathbf{}
        r'\\textit\{([^}]*)\}': r'\\mathit{\1}',  # \textit{} -> \mathit{}
        r'\\text\{([^}]*)\}': r'\\mathrm{\1}',  # \text{} -> \mathrm{}
    }
    
    result = latex_str
    for old, new in replacements.items():
        result = re.sub(old, new, result)
    
    return result

def render_latex_matplotlib(latex_str, output_base, width=8, height=2, fontsize=12, dpi=100):
    """
    Renders a LaTeX string using Matplotlib's math rendering and exports PNG + SVG.
    
    Args:
        latex_str: The LaTeX string to render
        output_base: Base path for output files without extension
        width: Figure width in inches
        height: Figure height in inches
        fontsize: Font size for the LaTeX text
        dpi: Resolution in dots per inch
    
    Returns:
        bool: Success status
    """
    try:
        # Create figure
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111)
        
        # Remove axes
        ax.axis('off')
        
        # Format the string for matplotlib's math mode
        latex_text = convert_latex_to_matplotlib(latex_str)
        
        # Ensure the text is in math mode (enclosed in $ symbols) if not already
        if '$' not in latex_text:
            latex_text = f"${latex_text}$"
        
        # Center the text in the figure
        ax.text(0.5, 0.5, latex_text, 
                fontsize=fontsize, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        
        # Adjust layout
        plt.tight_layout(pad=0.1)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_base), exist_ok=True)
        
        # Save as PNG and SVG
        plt.savefig(f"{output_base}.png", dpi=dpi, format='png', bbox_inches='tight')
        plt.savefig(f"{output_base}.svg", format='svg', bbox_inches='tight')
        
        # Close figure to free memory
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error rendering {output_base}: {str(e)}")
        return False

def extract_latex_formulas(text):
    """
    Extract LaTeX formulas from text using various patterns.
    
    Args:
        text: Text containing LaTeX formulas
        
    Returns:
        list: List of extracted formulas
    """
    formulas = []
    
    # Pattern 1: Display math - \[ ... \]
    pattern1 = r'\\[(.*?)\\]'
    for match in re.finditer(pattern1, text, re.DOTALL):
        formulas.append(match.group(1).strip())
    
    # Pattern 2: Inline math - $ ... $
    pattern2 = r'\$(.*?)\$'
    for match in re.finditer(pattern2, text):
        if match.group(1).strip() and '$' not in match.group(1):
            formulas.append(match.group(1).strip())
    
    # Pattern 3: LaTeX commands with arguments - e.g., \frac{a}{b}
    pattern3 = r'\\[a-zA-Z]+(\{[^{}]*\})+'
    for match in re.finditer(pattern3, text):
        if len(match.group(0)) > 3:  # Avoid trivial matches
            formulas.append(match.group(0).strip())
    
    # Remove duplicates while preserving order
    unique_formulas = []
    for formula in formulas:
        if formula not in unique_formulas:
            unique_formulas.append(formula)
    
    return unique_formulas

def render_full_statements(yaml_path, out_dir, width=8, height=3, fontsize=12, dpi=150, 
                          verbose=True, limit=None):
    """
    Render full LaTeX statements from a YAML file.
    
    Args:
        yaml_path: Path to the YAML file containing LaTeX statements
        out_dir: Output directory for the rendered images
        width: Figure width in inches
        height: Figure height in inches
        fontsize: Font size for the LaTeX text
        dpi: Resolution in dots per inch
        verbose: Print detailed progress information
        limit: Maximum number of problems to process (None for all)
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load YAML file
    with open(yaml_path, "r", encoding="utf-8") as f:
        problems = yaml.safe_load(f)
    
    # Apply limit if specified
    if limit is not None:
        problems = problems[:limit]
    
    success_count = 0
    total_count = len(problems)
    
    # Process each problem
    for i, item in enumerate(problems):
        try:
            name = item["problem_name"]
            stmt = item["informal_statement"]
            
            if verbose:
                print(f"Processing problem {i+1}/{total_count}: {name}")
            
            # Output path
            output_base = os.path.join(out_dir, name + "_stmt")
            
            # Render and export
            if render_latex_matplotlib(stmt, output_base, width, height, fontsize, dpi):
                success_count += 1
                if verbose:
                    print(f"  Exported: {output_base}.png and {output_base}.svg")
        except Exception as e:
            print(f"Error processing problem {i+1}: {str(e)}")
    
    print(f"\nCompleted: {success_count}/{total_count} problems processed successfully")

def extract_and_render_formulas(yaml_path, out_dir, width=8, height=2, fontsize=12, dpi=150, 
                               verbose=True, limit=None):
    """
    Extract math formulas from LaTeX text and render them separately.
    
    Args:
        yaml_path: Path to the YAML file containing LaTeX statements
        out_dir: Output directory for the rendered images
        width: Figure width in inches
        height: Figure height in inches
        fontsize: Font size for the LaTeX text
        dpi: Resolution in dots per inch
        verbose: Print detailed progress information
        limit: Maximum number of problems to process (None for all)
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load YAML file
    with open(yaml_path, "r", encoding="utf-8") as f:
        problems = yaml.safe_load(f)
    
    # Apply limit if specified
    if limit is not None:
        problems = problems[:limit]
    
    success_count = 0
    formula_count = 0
    total_problems_with_formulas = 0
    
    # Process each problem
    for i, item in enumerate(problems):
        try:
            name = item["problem_name"]
            stmt = item["informal_statement"]
            
            if verbose:
                print(f"Processing problem {i+1}/{len(problems)}: {name}")
            
            # Extract formulas
            formulas = extract_latex_formulas(stmt)
            
            if not formulas:
                if verbose:
                    print(f"  No formulas found in {name}")
                continue
            
            total_problems_with_formulas += 1
            formula_count += len(formulas)
            
            # Render each formula separately
            for j, formula in enumerate(formulas):
                output_base = os.path.join(out_dir, f"{name}_formula_{j+1}")
                
                if render_latex_matplotlib(formula, output_base, width, height, fontsize, dpi):
                    success_count += 1
                    if verbose:
                        print(f"  Exported formula {j+1}/{len(formulas)}: {output_base}")
                
        except Exception as e:
            print(f"Error processing problem {i+1}: {str(e)}")
    
    print(f"\nExtracted {formula_count} formulas from {total_problems_with_formulas}/{len(problems)} problems")
    print(f"Successfully rendered {success_count} formulas")

def main():
    parser = argparse.ArgumentParser(description="Render LaTeX statements or formulas from a YAML file using Matplotlib")
    parser.add_argument("--yaml", default="minimal_fork_of_putnambench_with_clear_answers.yaml",
                        help="Path to the YAML file containing LaTeX statements")
    parser.add_argument("--mode", choices=["full", "formulas", "both"], default="full",
                        help="Rendering mode: 'full' for full statements, 'formulas' for extracted formulas, 'both' for both")
    parser.add_argument("--out-dir", default="putnam_problems_images",
                        help="Output directory for rendered images")
    parser.add_argument("--formulas-dir", default="putnam_formulas_images",
                        help="Output directory for extracted formulas (when mode is 'formulas' or 'both')")
    parser.add_argument("--width", type=float, default=8.0,
                        help="Width of the output images in inches")
    parser.add_argument("--height", type=float, default=3.0,
                        help="Height of the output images in inches")
    parser.add_argument("--fontsize", type=int, default=12,
                        help="Font size for the LaTeX text")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Resolution in dots per inch")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of problems to process")
    
    args = parser.parse_args()
    
    # Validate YAML file
    if not os.path.exists(args.yaml):
        print(f"Error: YAML file '{args.yaml}' not found")
        return
    
    # Process according to mode
    if args.mode in ["full", "both"]:
        print(f"Rendering full statements from {args.yaml} to {args.out_dir}...")
        render_full_statements(args.yaml, args.out_dir, 
                              args.width, args.height, args.fontsize, args.dpi,
                              args.verbose, args.limit)
    
    if args.mode in ["formulas", "both"]:
        print(f"Extracting and rendering formulas from {args.yaml} to {args.formulas_dir}...")
        extract_and_render_formulas(args.yaml, args.formulas_dir,
                                   args.width, args.height, args.fontsize, args.dpi,
                                   args.verbose, args.limit)
    
    print("Done!")

if __name__ == "__main__":
    main() 