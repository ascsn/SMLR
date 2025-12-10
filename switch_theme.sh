#!/bin/bash
# =============================================================================
# SMLR Documentation Theme Switcher
# =============================================================================
# This script allows you to easily switch between documentation themes.
#
# Usage:
#   ./switch_theme.sh [theme]
#
# Available themes:
#   default       - Clean Material theme (original)
#   neobrutalism  - Bold, colorful neobrutalism style
#   status        - Show current theme
#
# Examples:
#   ./switch_theme.sh default       # Switch to default theme
#   ./switch_theme.sh neobrutalism  # Switch to neobrutalism theme
#   ./switch_theme.sh status        # Show which theme is active
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MKDOCS_FILE="$SCRIPT_DIR/mkdocs.yml"
CSS_DIR="$SCRIPT_DIR/docs/stylesheets"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${PURPLE}╔════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║   ${CYAN}SMLR Documentation Theme Switcher${PURPLE}   ║${NC}"
    echo -e "${PURPLE}╚════════════════════════════════════════╝${NC}"
    echo ""
}

print_usage() {
    echo -e "${YELLOW}Usage:${NC} $0 [theme]"
    echo ""
    echo -e "${YELLOW}Available themes:${NC}"
    echo -e "  ${GREEN}default${NC}       - Clean Material theme (original)"
    echo -e "  ${GREEN}neobrutalism${NC}  - Bold, colorful neobrutalism style"
    echo -e "  ${GREEN}status${NC}        - Show current theme"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 default       # Switch to default theme"
    echo "  $0 neobrutalism  # Switch to neobrutalism theme"
    echo "  $0 status        # Show which theme is active"
    echo ""
}

check_current_theme() {
    if grep -q "neobrutalism.css" "$MKDOCS_FILE" 2>/dev/null; then
        echo "neobrutalism"
    else
        echo "default"
    fi
}

switch_to_default() {
    echo -e "${BLUE}Switching to default theme...${NC}"
    
    # Check if backup exists
    if [ ! -f "$SCRIPT_DIR/mkdocs.default.yml" ]; then
        echo -e "${RED}Error: mkdocs.default.yml backup not found!${NC}"
        echo "Cannot switch to default theme without backup."
        exit 1
    fi
    
    # Restore the default mkdocs.yml
    cp "$SCRIPT_DIR/mkdocs.default.yml" "$MKDOCS_FILE"
    
    echo -e "${GREEN}✓ Switched to default theme!${NC}"
    echo -e "${CYAN}Run 'mkdocs serve' to preview the changes.${NC}"
}

switch_to_neobrutalism() {
    echo -e "${BLUE}Switching to neobrutalism theme...${NC}"
    
    # Check if neobrutalism CSS exists
    if [ ! -f "$CSS_DIR/neobrutalism.css" ]; then
        echo -e "${RED}Error: neobrutalism.css not found!${NC}"
        echo "Please ensure docs/stylesheets/neobrutalism.css exists."
        exit 1
    fi
    
    # Check if already using neobrutalism
    if grep -q "neobrutalism.css" "$MKDOCS_FILE"; then
        echo -e "${YELLOW}Already using neobrutalism theme.${NC}"
        return
    fi
    
    # Add neobrutalism.css to extra_css in mkdocs.yml
    if grep -q "extra_css:" "$MKDOCS_FILE"; then
        # Check if stylesheets/extra.css is the last entry
        if grep -q "stylesheets/extra.css" "$MKDOCS_FILE"; then
            # Add neobrutalism.css after extra.css
            sed -i.bak 's|  - stylesheets/extra.css|  - stylesheets/extra.css\n  - stylesheets/neobrutalism.css|' "$MKDOCS_FILE"
            rm -f "${MKDOCS_FILE}.bak"
        fi
    fi
    
    echo -e "${GREEN}✓ Switched to neobrutalism theme!${NC}"
    echo ""
    echo -e "${PURPLE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║${NC}  ${YELLOW}🎨 Neobrutalism Features:${NC}                   ${PURPLE}║${NC}"
    echo -e "${PURPLE}║${NC}  • Bold, high-contrast colors             ${PURPLE}║${NC}"
    echo -e "${PURPLE}║${NC}  • Hard offset shadows (no blur)          ${PURPLE}║${NC}"
    echo -e "${PURPLE}║${NC}  • Thick black borders                    ${PURPLE}║${NC}"
    echo -e "${PURPLE}║${NC}  • Vibrant accent colors                  ${PURPLE}║${NC}"
    echo -e "${PURPLE}║${NC}  • Playful, raw aesthetic                 ${PURPLE}║${NC}"
    echo -e "${PURPLE}╚════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Run 'mkdocs serve' to preview the changes.${NC}"
}

show_status() {
    current=$(check_current_theme)
    echo -e "${BLUE}Current theme:${NC} ${GREEN}$current${NC}"
    echo ""
    echo -e "${YELLOW}Available themes:${NC}"
    
    if [ "$current" = "default" ]; then
        echo -e "  ${GREEN}● default${NC} (active)"
        echo -e "  ${NC}○ neobrutalism${NC}"
    else
        echo -e "  ${NC}○ default${NC}"
        echo -e "  ${GREEN}● neobrutalism${NC} (active)"
    fi
    
    echo ""
    echo -e "${CYAN}Files:${NC}"
    echo -e "  mkdocs.yml:           $([ -f "$MKDOCS_FILE" ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
    echo -e "  mkdocs.default.yml:   $([ -f "$SCRIPT_DIR/mkdocs.default.yml" ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
    echo -e "  extra.css:            $([ -f "$CSS_DIR/extra.css" ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
    echo -e "  extra.default.css:    $([ -f "$CSS_DIR/extra.default.css" ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
    echo -e "  neobrutalism.css:     $([ -f "$CSS_DIR/neobrutalism.css" ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
}

# Main script
print_header

if [ $# -eq 0 ]; then
    print_usage
    show_status
    exit 0
fi

case "$1" in
    default)
        switch_to_default
        ;;
    neobrutalism|neo|brutal)
        switch_to_neobrutalism
        ;;
    status|--status|-s)
        show_status
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown theme: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
