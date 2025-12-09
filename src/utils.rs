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

/// A builder for creating formatted tables with Unicode box-drawing characters.
///
/// Supports both complete table generation and incremental streaming output.
/// Use streaming methods (`header()`, `row()`, `footer()`) when you need to
/// print rows incrementally, or use `build()` for a complete table.
///
/// Column widths can be specified explicitly or calculated automatically from content.
///
/// # Example
/// ```
/// use renoir::utils::TableBuilder;
///
/// // Explicit column widths
/// let table = TableBuilder::new(&[("Name", 10), ("Value", 8)]);
/// print!("{}", table.header());
/// print!("{}", table.row(&["foo", "42"]));
/// print!("{}", table.row(&["bar", "123"]));
/// print!("{}", table.footer());
///
/// // Auto-sized columns (width = 0 means auto-calculate)
/// let table = TableBuilder::new(&[("Name", 0), ("Value", 0)]);
/// let rows = vec![
///     vec!["foo", "42"],
///     vec!["bar", "123456"],
/// ];
/// print!("{}", table.build(&rows));
/// ```
pub struct TableBuilder {
    /// Column definitions: (header name, width)
    columns: Vec<(String, usize)>,
}

impl TableBuilder {
    /// Creates a new TableBuilder with the specified columns.
    ///
    /// If width is 0, it will be auto-calculated based on content when using `build()`.
    /// For streaming output (`header()`, `row()`, `footer()`), a minimum width based
    /// on the header name length + 2 will be used.
    ///
    /// # Arguments
    /// * `columns` - Slice of tuples containing (header_name, column_width)
    ///
    /// # Example
    /// ```
    /// use renoir::utils::TableBuilder;
    ///
    /// // Explicit widths
    /// let table = TableBuilder::new(&[
    ///     ("ID", 6),
    ///     ("Name", 20),
    ///     ("Score", 10),
    /// ]);
    ///
    /// // Auto-width (set to 0)
    /// let table = TableBuilder::new(&[
    ///     ("ID", 0),
    ///     ("Name", 0),
    /// ]);
    /// ```
    pub fn new(columns: &[(&str, usize)]) -> Self {
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
        }
    }

    /// Creates a new TableBuilder with auto-calculated column widths based on data.
    ///
    /// This analyzes the provided rows and calculates optimal widths for each column.
    ///
    /// # Arguments
    /// * `headers` - Slice of header names
    /// * `rows` - Slice of rows to analyze for width calculation
    ///
    /// # Example
    /// ```
    /// use renoir::utils::TableBuilder;
    ///
    /// let rows = vec![
    ///     vec!["foo", "42"],
    ///     vec!["barbaz", "123456"],
    /// ];
    /// let table = TableBuilder::from_data(&["Name", "Value"], &rows);
    /// print!("{}", table.build(&rows));
    /// ```
    pub fn from_data(headers: &[&str], rows: &[Vec<&str>]) -> Self {
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();

        // Find the max width for each column from row data
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        // Add padding (2 characters on each side)
        let columns: Vec<(String, usize)> = headers
            .iter()
            .zip(widths.iter())
            .map(|(name, width)| (name.to_string(), width + 2))
            .collect();

        Self { columns }
    }

    /// Updates column widths based on the provided rows.
    ///
    /// This is useful when you want to calculate widths before streaming output.
    ///
    /// # Arguments
    /// * `rows` - Slice of rows to analyze for width calculation
    ///
    /// # Returns
    /// A new `TableBuilder` with updated widths.
    pub fn with_data(self, rows: &[Vec<&str>]) -> Self {
        let mut widths: Vec<usize> = self.columns.iter().map(|(name, w)| (*w).max(name.len())).collect();

        // Find the max width for each column from row data
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len() + 2); // Add padding
                }
            }
        }

        Self {
            columns: self
                .columns
                .iter()
                .zip(widths.iter())
                .map(|((name, _), width)| (name.clone(), *width))
                .collect(),
        }
    }

    /// Generates the table header (top border + header row + separator).
    ///
    /// # Returns
    /// A `String` containing the header portion of the table.
    ///
    /// # Example
    /// ```
    /// use renoir::utils::TableBuilder;
    ///
    /// let table = TableBuilder::new(&[("Name", 10), ("Value", 8)]);
    /// print!("{}", table.header());
    /// // ┌──────────┬────────┐
    /// // │   Name   │ Value  │
    /// // ├──────────┼────────┤
    /// ```
    pub fn header(&self) -> String {
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

    /// Generates a single data row.
    ///
    /// Cell values are centered within their column width.
    ///
    /// # Arguments
    /// * `cells` - Slice of cell values (should match the number of columns)
    ///
    /// # Returns
    /// A `String` containing a single table row.
    ///
    /// # Example
    /// ```
    /// use renoir::utils::TableBuilder;
    ///
    /// let table = TableBuilder::new(&[("Name", 10), ("Value", 8)]);
    /// print!("{}", table.row(&["foo", "42"]));
    /// // │   foo    │   42   │
    /// ```
    pub fn row(&self, cells: &[&str]) -> String {
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

    /// Generates the table footer (bottom border).
    ///
    /// # Returns
    /// A `String` containing the footer portion of the table.
    ///
    /// # Example
    /// ```
    /// use renoir::utils::TableBuilder;
    ///
    /// let table = TableBuilder::new(&[("Name", 10), ("Value", 8)]);
    /// print!("{}", table.footer());
    /// // └──────────┴────────┘
    /// ```
    pub fn footer(&self) -> String {
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

    /// Generates a complete table with all rows.
    ///
    /// This is a convenience method that combines `header()`, multiple `row()` calls,
    /// and `footer()` into a single string.
    ///
    /// # Arguments
    /// * `rows` - Slice of rows, where each row is a Vec of cell values
    ///
    /// # Returns
    /// A `String` containing the complete table.
    ///
    /// # Example
    /// ```
    /// use renoir::utils::TableBuilder;
    ///
    /// let table = TableBuilder::new(&[("Name", 10), ("Value", 8)]);
    /// let rows = vec![
    ///     vec!["foo", "42"],
    ///     vec!["bar", "123"],
    /// ];
    /// print!("{}", table.build(&rows));
    /// ```
    pub fn build(&self, rows: &[Vec<&str>]) -> String {
        let mut result = self.header();
        for row in rows {
            result.push_str(&self.row(row));
        }
        result.push_str(&self.footer());
        result
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
    fn test_table_builder() {
        let table = TableBuilder::new(&[("A", 5), ("B", 5)]);
        let header = table.header();
        assert!(header.contains("┌"));
        assert!(header.contains("A"));
        assert!(header.contains("B"));

        let row = table.row(&["1", "2"]);
        assert!(row.contains("1"));
        assert!(row.contains("2"));

        let footer = table.footer();
        assert!(footer.contains("└"));
    }

    #[test]
    fn test_table_build_complete() {
        let table = TableBuilder::new(&[("X", 5), ("Y", 5)]);
        let rows = vec![vec!["a", "b"], vec!["c", "d"]];
        let complete = table.build(&rows);
        assert!(complete.contains("┌"));
        assert!(complete.contains("└"));
        assert!(complete.contains("a"));
        assert!(complete.contains("d"));
    }
}

