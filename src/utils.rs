//! # Formatting Utilities
//!
//! This module provides utilities for consistent console output formatting,
//! including banners and tables using Unicode box-drawing characters.
//!
//! ## Examples
//!
//! ### Simple Banner
//! ```
//! use renoir::utils::create_banner;
//!
//! let banner = create_banner("MY TITLE", &[]);
//! println!("{}", banner);
//! // ╔══════════════════════════════════════════════════════════════╗
//! // ║                          MY TITLE                            ║
//! // ╚══════════════════════════════════════════════════════════════╝
//! ```
//!
//! ### Banner with Content
//! ```
//! use renoir::utils::create_banner;
//!
//! let banner = create_banner("SUMMARY", &["Total: 10", "Average: 5.5"]);
//! println!("{}", banner);
//! // ╔══════════════════════════════════════════════════════════════╗
//! // ║                          SUMMARY                             ║
//! // ╠══════════════════════════════════════════════════════════════╣
//! // ║  Total: 10                                                   ║
//! // ║  Average: 5.5                                                ║
//! // ╚══════════════════════════════════════════════════════════════╝
//! ```
//!
//! ### Table with Streaming Output
//! ```
//! use renoir::utils::TableBuilder;
//!
//! let table = TableBuilder::new(&[("Name", 10), ("Value", 8)]);
//! print!("{}", table.header());
//! print!("{}", table.row(&["foo", "42"]));
//! print!("{}", table.row(&["bar", "123"]));
//! print!("{}", table.footer());
//! ```

/// Default width for banners (including border characters).
pub const DEFAULT_BANNER_WIDTH: usize = 64;

/// Minimum width for banners.
pub const MIN_BANNER_WIDTH: usize = 20;

/// Creates a banner with a title and optional content lines.
///
/// The banner uses Unicode box-drawing characters. Width is automatically
/// calculated based on the longest content line, with a minimum of 20 characters
/// and a default/minimum of 64 characters for aesthetic purposes.
///
/// # Arguments
/// * `title` - The banner title (centered)
/// * `content_lines` - Optional pre-formatted content lines to display below the title
///
/// # Returns
/// A `String` containing the complete banner ready to be printed.
///
/// # Example
/// ```
/// use renoir::utils::create_banner;
///
/// // Simple banner
/// let banner = create_banner("MY TITLE", &[]);
/// println!("{}", banner);
///
/// // Banner with content (width adjusts to fit)
/// let banner = create_banner("SUMMARY", &[
///     &format!("Total: {}", 10),
///     &format!("Average: {:.2}", 5.5),
/// ]);
/// println!("{}", banner);
/// ```
pub fn create_banner(title: &str, content_lines: &[&str]) -> String {
    // Calculate the required width based on content
    let title_width = title.len() + 4; // Add some padding around the title
    let content_max_width = content_lines
        .iter()
        .map(|line| line.len() + 4) // Add indent (2) + padding (2)
        .max()
        .unwrap_or(0);

    // Use the maximum of: default width, title width, or content width
    let inner_width = DEFAULT_BANNER_WIDTH
        .max(title_width)
        .max(content_max_width)
        .max(MIN_BANNER_WIDTH)
        - 2; // Subtract 2 for the border characters

    let mut result = String::new();

    // Top border
    result.push_str("\n╔");
    result.push_str(&"═".repeat(inner_width));
    result.push_str("╗\n");

    // Title line (centered)
    let title_padding = inner_width.saturating_sub(title.len());
    let left_pad = title_padding / 2;
    let right_pad = title_padding - left_pad;
    result.push_str("║");
    result.push_str(&" ".repeat(left_pad));
    result.push_str(title);
    result.push_str(&" ".repeat(right_pad));
    result.push_str("║\n");

    // If there are content lines, add separator and content
    if !content_lines.is_empty() {
        // Separator
        result.push_str("╠");
        result.push_str(&"═".repeat(inner_width));
        result.push_str("╣\n");

        // Content lines (left-aligned with 2-space indent)
        for line in content_lines {
            let content = format!("  {}", line);
            let padding = inner_width.saturating_sub(content.len());
            result.push_str("║");
            result.push_str(&content);
            result.push_str(&" ".repeat(padding));
            result.push_str("║\n");
        }
    }

    // Bottom border
    result.push_str("╚");
    result.push_str(&"═".repeat(inner_width));
    result.push_str("╝\n");

    result
}

/// A stateful table that automatically prints rows with separators.
///
/// This struct handles formatted table output with Unicode box-drawing characters.
/// It automatically:
/// - Prints the header when the first row is added
/// - Prints separators between rows
/// - Prints the footer when [`finish()`](Table::finish) is called or on drop
///
/// # Example: Batch row addition
/// ```
/// use renoir::utils::Table;
///
/// let mut table = Table::new(&[("Name", 10), ("Value", 8)]);
/// table.add_row(&["foo", "42"]);
/// table.add_row(&["bar", "123"]);
/// table.finish();
/// ```
///
/// # Example: Incremental cell population
/// ```
/// use renoir::utils::Table;
///
/// let mut table = Table::new(&[("Name", 10), ("Value", 8)]);
/// table.set_cell(0, "foo");
/// table.set_cell(1, "42");
/// table.flush_row();  // Finalizes the row
/// table.start_row();
/// table.set_cell(0, "bar");  // Updates display immediately
/// table.set_cell(1, "123");
/// table.flush_row();
/// table.finish();
/// ```
pub struct Table {
    /// Column definitions: (header name, width)
    columns: Vec<(String, usize)>,
    /// Buffer for the current row's cells (for incremental population)
    cell_buffer: Vec<String>,
    current_row: usize,
    header_printed: bool,
    row_started: bool,
    finished: bool,
}

impl Table {
    /// Creates a new Table with the specified columns.
    ///
    /// # Arguments
    /// * `columns` - Slice of tuples containing (header_name, column_width)
    ///
    /// # Example
    /// ```
    /// use renoir::utils::Table;
    ///
    /// let mut table = Table::new(&[("ID", 6), ("Name", 20), ("Score", 10)]);
    /// ```
    pub fn new(columns: &[(&str, usize)]) -> Self {
        let num_cols = columns.len();
        Self {
            columns: columns
                .iter()
                .map(|(name, width)| {
                    // If the width is 0, use header length + 2 as the minimum
                    let effective_width = if *width == 0 {
                        name.len() + 2
                    } else {
                        *width
                    };
                    (name.to_string(), effective_width)
                })
                .collect(),
            cell_buffer: vec![String::new(); num_cols],
            current_row: 0,
            header_printed: false,
            row_started: false,
            finished: false,
        }
    }

    /// Starts a new row, printing header/separator as needed and showing empty cells.
    ///
    /// Call this before setting cells for a new row. The row is displayed immediately
    /// with empty cells, then `set_cell()` updates cells in real-time.
    pub fn start_row(&mut self) {
        use std::io::{self, Write};
        
        // Print header on first row
        if !self.header_printed {
            print!("{}", self.format_header());
            self.header_printed = true;
        } else if self.current_row > 0 {
            // Print separator before this row (not before first data row)
            print!("{}", self.format_separator());
        }
        
        // Clear buffer for new row
        for cell in &mut self.cell_buffer {
            cell.clear();
        }
        
        // Print empty row (without newline - we'll overwrite it)
        self.print_current_row_inline();
        io::stdout().flush().ok();
        self.row_started = true;
    }

    /// Sets a cell value and immediately updates the display.
    ///
    /// The row is reprinted in-place using carriage return, showing the updated cell.
    ///
    /// # Arguments
    /// * `col` - Column index (0-based)
    /// * `value` - Cell value to set
    pub fn set_cell(&mut self, col: usize, value: &str) {
        use std::io::{self, Write};
        
        if col < self.cell_buffer.len() {
            self.cell_buffer[col] = value.to_string();
            
            // If row has started, update the display
            if self.row_started {
                self.print_current_row_inline();
                io::stdout().flush().ok();
            }
        }
    }

    /// Finalizes the current row with a newline and advances to the next row.
    pub fn flush_row(&mut self) {
        if self.row_started {
            println!();  // End the current row with newline
            self.current_row += 1;
            self.row_started = false;
        }
    }

    /// Prints the current row inline (without newline) for live updates.
    fn print_current_row_inline(&self) {
        let cells: Vec<&str> = self.cell_buffer.iter().map(|s| s.as_str()).collect();
        let row_str = self.format_row_no_newline(&cells);
        print!("\r{}", row_str);
    }

    /// Generates a single data row WITHOUT trailing newline (for live updates).
    fn format_row_no_newline(&self, cells: &[&str]) -> String {
        let mut result = String::new();
        result.push('│');

        for (i, (_, width)) in self.columns.iter().enumerate() {
            let cell = cells.get(i).copied().unwrap_or("");
            let padding = width.saturating_sub(cell.len());
            let left_pad = padding / 2;
            let right_pad = padding - left_pad;
            result.push_str(&" ".repeat(left_pad));
            result.push_str(cell);
            result.push_str(&" ".repeat(right_pad));
            if i < self.columns.len() - 1 {
                result.push('│');
            }
        }
        result.push('│');

        result
    }


    /// Adds and immediately prints a row to the table.
    ///
    /// On the first call, this also prints the table header.
    /// Between rows, this automatically prints separators.
    ///
    /// # Arguments
    /// * `cells` - Slice of cell values (should match the number of columns)
    pub fn add_row(&mut self, cells: &[&str]) {
        // Print header on first row
        if !self.header_printed {
            print!("{}", self.format_header());
            self.header_printed = true;
        } else {
            // Print separator before this row (after previous row)
            print!("{}", self.format_separator());
        }

        // Print the row
        print!("{}", self.format_row(cells));
        self.current_row += 1;
    }

    /// Finishes the table by printing the footer.
    ///
    /// This is called automatically on drop, but can be called explicitly
    /// if you need the footer printed at a specific point.
    pub fn finish(&mut self) {
        if !self.finished && self.header_printed {
            print!("{}", self.format_footer());
            self.finished = true;
        }
    }


    // ========================================================================
    // Internal formatting methods
    // ========================================================================

    /// Generates the table header (top border + header row + separator).
    fn format_header(&self) -> String {
        let mut result = String::new();

        // Top border: ┌────┬────┬────┐
        result.push('┌');
        for (i, (_, width)) in self.columns.iter().enumerate() {
            result.push_str(&"─".repeat(*width));
            if i < self.columns.len() - 1 {
                result.push('┬');
            }
        }
        result.push_str("┐\n");

        // Header row: │ Name │ Value │
        result.push('│');
        for (i, (name, width)) in self.columns.iter().enumerate() {
            let padding = width.saturating_sub(name.len());
            let left_pad = padding / 2;
            let right_pad = padding - left_pad;
            result.push_str(&" ".repeat(left_pad));
            result.push_str(name);
            result.push_str(&" ".repeat(right_pad));
            if i < self.columns.len() - 1 {
                result.push('│');
            }
        }
        result.push_str("│\n");

        // Separator: ├────┼────┼────┤
        result.push('├');
        for (i, (_, width)) in self.columns.iter().enumerate() {
            result.push_str(&"─".repeat(*width));
            if i < self.columns.len() - 1 {
                result.push('┼');
            }
        }
        result.push_str("┤\n");

        result
    }

    /// Generates a single data row with centered cell values.
    fn format_row(&self, cells: &[&str]) -> String {
        let mut result = String::new();
        result.push('│');

        for (i, (_, width)) in self.columns.iter().enumerate() {
            let cell = cells.get(i).copied().unwrap_or("");
            let padding = width.saturating_sub(cell.len());
            let left_pad = padding / 2;
            let right_pad = padding - left_pad;
            result.push_str(&" ".repeat(left_pad));
            result.push_str(cell);
            result.push_str(&" ".repeat(right_pad));
            if i < self.columns.len() - 1 {
                result.push('│');
            }
        }
        result.push_str("│\n");

        result
    }

    /// Generates a row separator (horizontal line between rows).
    fn format_separator(&self) -> String {
        let mut result = String::new();

        // Separator: ├────┼────┼────┤
        result.push('├');
        for (i, (_, width)) in self.columns.iter().enumerate() {
            result.push_str(&"─".repeat(*width));
            if i < self.columns.len() - 1 {
                result.push('┼');
            }
        }
        result.push_str("┤\n");

        result
    }

    /// Generates the table footer (bottom border).
    fn format_footer(&self) -> String {
        let mut result = String::new();

        // Bottom border: └────┴────┴────┘
        result.push('└');
        for (i, (_, width)) in self.columns.iter().enumerate() {
            result.push_str(&"─".repeat(*width));
            if i < self.columns.len() - 1 {
                result.push('┴');
            }
        }
        result.push_str("┘\n");

        result
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_banner_simple() {
        let banner = create_banner("TEST", &[]);
        assert!(banner.contains("TEST"));
        assert!(banner.contains("╔"));
        assert!(banner.contains("╚"));
    }

    #[test]
    fn test_banner_with_content() {
        let banner = create_banner("SUMMARY", &["Line 1", "Line 2"]);
        assert!(banner.contains("SUMMARY"));
        assert!(banner.contains("Line 1"));
        assert!(banner.contains("Line 2"));
        assert!(banner.contains("╠")); // Separator should be present
    }

    #[test]
    fn test_table_formatting() {
        let table = Table::new(&[("A", 5), ("B", 5)]);
        let header = table.format_header();
        assert!(header.contains("┌"));
        assert!(header.contains("A"));
        assert!(header.contains("B"));

        let row = table.format_row(&["1", "2"]);
        assert!(row.contains("1"));
        assert!(row.contains("2"));

        let footer = table.format_footer();
        assert!(footer.contains("└"));
    }

    #[test]
    fn test_table_separator() {
        let table = Table::new(&[("X", 5), ("Y", 5)]);
        let sep = table.format_separator();
        assert!(sep.contains("├"));
        assert!(sep.contains("┼"));
        assert!(sep.contains("┤"));
    }
}

