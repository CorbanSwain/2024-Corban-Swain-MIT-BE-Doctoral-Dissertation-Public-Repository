<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="21008000">
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="classes" Type="Folder"/>
		<Item Name="controls" Type="Folder">
			<Property Name="NI.SortType" Type="Int">0</Property>
			<Item Name="flat enum.ctl" Type="VI" URL="../controls/flat enum.ctl"/>
			<Item Name="transparent_enum.ctl" Type="VI" URL="../controls/transparent_enum.ctl"/>
		</Item>
		<Item Name="sandbox-vis" Type="Folder">
			<Property Name="NI.SortType" Type="Int">0</Property>
			<Item Name="sensor integration for debug async v5.vi" Type="VI" URL="../sensor integration for debug async v5.vi"/>
			<Item Name="sensor_integration_v2.vi" Type="VI" URL="../sensor_integration_v2.vi"/>
			<Item Name="sensor_integration_v4.vi" Type="VI" URL="../sensor_integration_v4.vi"/>
			<Item Name="test to disable enum values.vi" Type="VI" URL="../test to disable enum values.vi"/>
			<Item Name="testing odor and manifold config dialog.vi" Type="VI" URL="../testing odor and manifold config dialog.vi"/>
			<Item Name="testing odor and manifold configuration.vi" Type="VI" URL="../testing odor and manifold configuration.vi"/>
			<Item Name="testing odor and manifold configuration_v3.vi" Type="VI" URL="../testing odor and manifold configuration_v3.vi"/>
			<Item Name="testing odor and manifold configuration_v4.vi" Type="VI" URL="../testing odor and manifold configuration_v4.vi"/>
			<Item Name="testing odor-manifold-stim config v2.vi" Type="VI" URL="../testing odor-manifold-stim config v2.vi"/>
			<Item Name="testing stimulator configuration_v2.vi" Type="VI" URL="../testing stimulator configuration_v2.vi"/>
			<Item Name="testing StringAndValues[] ring update.vi" Type="VI" URL="../testing StringAndValues[] ring update.vi"/>
		</Item>
		<Item Name="sub-vis" Type="Folder">
			<Item Name="arduino-sensirion" Type="Folder">
				<Property Name="NI.SortType" Type="Int">0</Property>
				<Item Name="arduino sensirion read to stream and log (sub-vi).vi" Type="VI" URL="../sub-vis/arduino sensirion read to stream and log (sub-vi).vi"/>
				<Item Name="arduino sensirion read with retry (sub-vi).vi" Type="VI" URL="../sub-vis/arduino sensirion read with retry (sub-vi).vi"/>
				<Item Name="arduino sensirion serial setup (sub-vi).vi" Type="VI" URL="../sub-vis/arduino sensirion serial setup (sub-vi).vi"/>
				<Item Name="check arduino parse error (sub-vi).vi" Type="VI" URL="../sub-vis/check arduino parse error (sub-vi).vi"/>
				<Item Name="clear arduino parse error (sub-vi).vi" Type="VI" URL="../sub-vis/clear arduino parse error (sub-vi).vi"/>
				<Item Name="generate sensorion data log header.vi" Type="VI" URL="../sub-vis/generate sensorion data log header.vi"/>
				<Item Name="parse_arduino_data_message (sub-vi).vi" Type="VI" URL="../sub-vis/parse_arduino_data_message (sub-vi).vi"/>
				<Item Name="read_sensirion_from_arduino (sub-vi).vi" Type="VI" URL="../sub-vis/read_sensirion_from_arduino (sub-vi).vi"/>
				<Item Name="sensirion data array to logstring (sub-vi).vi" Type="VI" URL="../sub-vis/sensirion data array to logstring (sub-vi).vi"/>
			</Item>
			<Item Name="general utilities" Type="Folder">
				<Property Name="NI.SortType" Type="Int">0</Property>
				<Item Name="append line ending (sub-vi).vi" Type="VI" URL="../sub-vis/append line ending (sub-vi).vi"/>
				<Item Name="bool to disabled state (sub-vi).vi" Type="VI" URL="../sub-vis/bool to disabled state (sub-vi).vi"/>
				<Item Name="bool to string (sub-vi).vi" Type="VI" URL="../sub-vis/bool to string (sub-vi).vi"/>
				<Item Name="convert custom error ring to error code (sub-vi).vi" Type="VI" URL="../sub-vis/convert custom error ring to error code (sub-vi).vi"/>
				<Item Name="double_arr_to_bool_arr (sub-vi).vi" Type="VI" URL="../sub-vis/double_arr_to_bool_arr (sub-vi).vi"/>
				<Item Name="Share Stop Signal (sub-vi).vi" Type="VI" URL="../sub-vis/Share Stop Signal (sub-vi).vi"/>
				<Item Name="xy chart buffer (subVI).vi" Type="VI" URL="../sub-vis/xy chart buffer (subVI).vi"/>
			</Item>
			<Item Name="logging" Type="Folder">
				<Property Name="NI.SortType" Type="Int">0</Property>
				<Item Name="append lines to loggfile (sub-vi).vi" Type="VI" URL="../sub-vis/append lines to loggfile (sub-vi).vi"/>
			</Item>
			<Item Name="odor-manifold config" Type="Folder">
				<Property Name="NI.SortType" Type="Int">0</Property>
				<Item Name="check manifold config array value errors (sub-vi).vi" Type="VI" URL="../sub-vis/check manifold config array value errors (sub-vi).vi"/>
				<Item Name="check odor config array value errors (sub-vi).vi" Type="VI" URL="../sub-vis/check odor config array value errors (sub-vi).vi"/>
				<Item Name="create odor and manifold configuration (sub-vi).vi" Type="VI" URL="../sub-vis/create odor and manifold configuration (sub-vi).vi"/>
				<Item Name="init odor config color ring (sub-vi).vi" Type="VI" URL="../sub-vis/init odor config color ring (sub-vi).vi"/>
				<Item Name="init odor descrip array (sub-vi).vi" Type="VI" URL="../sub-vis/init odor descrip array (sub-vi).vi"/>
				<Item Name="initialize manifold config array control (sub-vi).vi" Type="VI" URL="../sub-vis/initialize manifold config array control (sub-vi).vi"/>
				<Item Name="manifold config control active odor check (sub-vi).vi" Type="VI" URL="../sub-vis/manifold config control active odor check (sub-vi).vi"/>
				<Item Name="manifold config control duplicate check (sub-vi).vi" Type="VI" URL="../sub-vis/manifold config control duplicate check (sub-vi).vi"/>
				<Item Name="manifold config error (sub-vi).vi" Type="VI" URL="../sub-vis/manifold config error (sub-vi).vi"/>
				<Item Name="manifold config error box update (sub-vi).vi" Type="VI" URL="../sub-vis/manifold config error box update (sub-vi).vi"/>
				<Item Name="manifold config to odor count map (sub-vi).vi" Type="VI" URL="../sub-vis/manifold config to odor count map (sub-vi).vi"/>
				<Item Name="manifold config to odor set (sub-vi).vi" Type="VI" URL="../sub-vis/manifold config to odor set (sub-vi).vi"/>
				<Item Name="odor config arr color update (sub-vi).vi" Type="VI" URL="../sub-vis/odor config arr color update (sub-vi).vi"/>
				<Item Name="odor config arr descrip update (sub-vi).vi" Type="VI" URL="../sub-vis/odor config arr descrip update (sub-vi).vi"/>
				<Item Name="odor config arr to active odor set (sub-vi).vi" Type="VI" URL="../sub-vis/odor config arr to active odor set (sub-vi).vi"/>
				<Item Name="odor config arr to disabled odors array (sub-vi).vi" Type="VI" URL="../sub-vis/odor config arr to disabled odors array (sub-vi).vi"/>
				<Item Name="odor config array property control (sub-vi).vi" Type="VI" URL="../sub-vis/odor config array property control (sub-vi).vi"/>
				<Item Name="odor config array to odor-color map (sub-vi).vi" Type="VI" URL="../sub-vis/odor config array to odor-color map (sub-vi).vi"/>
				<Item Name="odor config array to odor-descrip map (sub-vi).vi" Type="VI" URL="../sub-vis/odor config array to odor-descrip map (sub-vi).vi"/>
				<Item Name="odor config error (sub-vi).vi" Type="VI" URL="../sub-vis/odor config error (sub-vi).vi"/>
				<Item Name="stim config control color update (sub-vi).vi" Type="VI" URL="../sub-vis/stim config control color update (sub-vi).vi"/>
				<Item Name="update stimulator config control (sub-vi).vi" Type="VI" URL="../sub-vis/update stimulator config control (sub-vi).vi"/>
			</Item>
			<Item Name="stimulation config" Type="Folder">
				<Property Name="NI.SortType" Type="Int">0</Property>
				<Item Name="check and update stimulator odor durations (sub-vi).vi" Type="VI" URL="../sub-vis/check and update stimulator odor durations (sub-vi).vi"/>
				<Item Name="configure stimulation dialog (subVI).vi" Type="VI" URL="../sub-vis/configure stimulation dialog (subVI).vi"/>
				<Item Name="detect changed entries in stim config array (sub-vi).vi" Type="VI" URL="../sub-vis/detect changed entries in stim config array (sub-vi).vi"/>
				<Item Name="odor-manifold config to port-odor ring entry (sub-vi).vi" Type="VI" URL="../sub-vis/odor-manifold config to port-odor ring entry (sub-vi).vi"/>
				<Item Name="port-odor cluster refnum split (sub-vi).vi" Type="VI" URL="../sub-vis/port-odor cluster refnum split (sub-vi).vi"/>
				<Item Name="reset port-odor ring StringsAndValues[] (sub-vi).vi" Type="VI" URL="../sub-vis/reset port-odor ring StringsAndValues[] (sub-vi).vi"/>
				<Item Name="reset port-odor ring value (sub-vi).vi" Type="VI" URL="../sub-vis/reset port-odor ring value (sub-vi).vi"/>
				<Item Name="stim config array refnum split (sub-vi).vi" Type="VI" URL="../sub-vis/stim config array refnum split (sub-vi).vi"/>
				<Item Name="stim config negative duration warning (sub-vi).vi" Type="VI" URL="../sub-vis/stim config negative duration warning (sub-vi).vi"/>
				<Item Name="stim line entry cluster refnum split (sub-vi).vi" Type="VI" URL="../sub-vis/stim line entry cluster refnum split (sub-vi).vi"/>
				<Item Name="update stim config control (sub-vi).vi" Type="VI" URL="../update stim config control (sub-vi).vi"/>
				<Item Name="update stimulator odor colors (sub-vi).vi" Type="VI" URL="../sub-vis/update stimulator odor colors (sub-vi).vi"/>
				<Item Name="update stimulator ring entries (sub-vi).vi" Type="VI" URL="../sub-vis/update stimulator ring entries (sub-vi).vi"/>
			</Item>
		</Item>
		<Item Name="type-defs" Type="Folder">
			<Item Name="arduino-sensirion" Type="Folder">
				<Property Name="NI.SortType" Type="Int">0</Property>
				<Item Name="ardruino_sensirion_read_states.ctl" Type="VI" URL="../type-defs/ardruino_sensirion_read_states.ctl"/>
				<Item Name="sensirion_data_with_time.ctl" Type="VI" URL="../type-defs/sensirion_data_with_time.ctl"/>
				<Item Name="sensirion_flow_sensor_data.ctl" Type="VI" URL="../controls/sensirion_flow_sensor_data.ctl"/>
			</Item>
			<Item Name="odor-manifold" Type="Folder">
				<Property Name="NI.SortType" Type="Int">0</Property>
				<Item Name="manifold configuration cluster (type def).ctl" Type="VI" URL="../type-defs/manifold configuration cluster (type def).ctl"/>
				<Item Name="manifold configuration cluster array (type def).ctl" Type="VI" URL="../type-defs/manifold configuration cluster array (type def).ctl"/>
				<Item Name="manifold selection.ctl" Type="VI" URL="../type-defs/manifold selection.ctl"/>
				<Item Name="odor and manifold config cluster (type def).ctl" Type="VI" URL="../type-defs/odor and manifold config cluster (type def).ctl"/>
				<Item Name="odor and manifold config UI states (type def).ctl" Type="VI" URL="../type-defs/odor and manifold config UI states (type def).ctl"/>
				<Item Name="odor channel to odor color map (type def).ctl" Type="VI" URL="../type-defs/odor channel to odor color map (type def).ctl"/>
				<Item Name="odor channel to odor count map (type def).ctl" Type="VI" URL="../type-defs/odor channel to odor count map (type def).ctl"/>
				<Item Name="odor channel to string map (type def).ctl" Type="VI" URL="../type-defs/odor channel to string map (type def).ctl"/>
				<Item Name="odor color (type def).ctl" Type="VI" URL="../type-defs/odor color (type def).ctl"/>
				<Item Name="odor color enum (type def).ctl" Type="VI" URL="../type-defs/odor color enum (type def).ctl"/>
				<Item Name="odor description array (type def).ctl" Type="VI" URL="../type-defs/odor description array (type def).ctl"/>
				<Item Name="odor info cluster (type def).ctl" Type="VI" URL="../type-defs/odor info cluster (type def).ctl"/>
				<Item Name="odor selection (type def).ctl" Type="VI" URL="../type-defs/odor selection (type def).ctl"/>
				<Item Name="port number (type def).ctl" Type="VI" URL="../type-defs/port number (type def).ctl"/>
				<Item Name="single port single odor cluster.ctl" Type="VI" URL="../type-defs/single port single odor cluster.ctl"/>
			</Item>
			<Item Name="stimulation" Type="Folder">
				<Property Name="NI.SortType" Type="Int">0</Property>
				<Item Name="port-odor ring (type def).ctl" Type="VI" URL="../type-defs/port-odor ring (type def).ctl"/>
				<Item Name="port-odor with color cluster (type def).ctl" Type="VI" URL="../type-defs/port-odor with color cluster (type def).ctl"/>
				<Item Name="stim. config array and port-odor refnums cluster.ctl" Type="VI" URL="../type-defs/stim. config array and port-odor refnums cluster.ctl"/>
				<Item Name="stimulation config UI states (type def).ctl" Type="VI" URL="../type-defs/stimulation config UI states (type def).ctl"/>
				<Item Name="stimulation configuration array (type def).ctl" Type="VI" URL="../type-defs/stimulation configuration array (type def).ctl"/>
				<Item Name="stimulator configuration line cluster (type def).ctl" Type="VI" URL="../type-defs/stimulator configuration line cluster (type def).ctl"/>
			</Item>
			<Item Name="bool_array_with_check.ctl" Type="VI" URL="../controls/bool_array_with_check.ctl"/>
			<Item Name="double_array_with_check.ctl" Type="VI" URL="../double_array_with_check.ctl"/>
			<Item Name="project_custom_error_codes_ring (type def).ctl" Type="VI" URL="../type-defs/project_custom_error_codes_ring (type def).ctl"/>
			<Item Name="ragged ring entry 2D array cluster (typed def).ctl" Type="VI" URL="../type-defs/ragged ring entry 2D array cluster (typed def).ctl"/>
			<Item Name="ring string and value cluster (type def).ctl" Type="VI" URL="../type-defs/ring string and value cluster (type def).ctl"/>
		</Item>
		<Item Name="arduino sensirion read to stream and log v2 (sub-vi).vi" Type="VI" URL="../sub-vis/arduino sensirion read to stream and log v2 (sub-vi).vi"/>
		<Item Name="bool 2d array to uint array.vi" Type="VI" URL="../sub-vis/bool 2d array to uint array.vi"/>
		<Item Name="cam_2p_status_logging_states.ctl" Type="VI" URL="../type-defs/cam_2p_status_logging_states.ctl"/>
		<Item Name="corrected button.ctl" Type="VI" URL="../controls/corrected button.ctl"/>
		<Item Name="create status logfile line (subVI).vi" Type="VI" URL="../sub-vis/create status logfile line (subVI).vi"/>
		<Item Name="create stim logfile line (subVI).vi" Type="VI" URL="../sub-vis/create stim logfile line (subVI).vi"/>
		<Item Name="filter_array (subVI).vi" Type="VI" URL="../sub-vis/filter_array (subVI).vi"/>
		<Item Name="flow rate chart 1.ctl" Type="VI" URL="../controls/flow rate chart 1.ctl"/>
		<Item Name="flow rate timeseries data preprocessing (subVI).vi" Type="VI" URL="../sub-vis/flow rate timeseries data preprocessing (subVI).vi"/>
		<Item Name="get and update stim ring entries (subVI).vi" Type="VI" URL="../sub-vis/get and update stim ring entries (subVI).vi"/>
		<Item Name="get valid conv subarray (subVI).vi" Type="VI" URL="../sub-vis/get valid conv subarray (subVI).vi"/>
		<Item Name="global variables.vi" Type="VI" URL="../global-variables/global variables.vi"/>
		<Item Name="handle input dir with subdir (subVI).vi" Type="VI" URL="../sub-vis/handle input dir with subdir (subVI).vi"/>
		<Item Name="init mc listbox stimulation status (subVI).vi" Type="VI" URL="../sub-vis/init mc listbox stimulation status (subVI).vi"/>
		<Item Name="maim stimulation ui states.ctl" Type="VI" URL="../type-defs/maim stimulation ui states.ctl"/>
		<Item Name="mc listbox set active cell (SubVI).vi" Type="VI" URL="../sub-vis/mc listbox set active cell (SubVI).vi"/>
		<Item Name="moving average.vi" Type="VI" URL="../sub-vis/moving average.vi"/>
		<Item Name="odor manifold config to csv (top level vi).vi" Type="VI" URL="../odor manifold config to csv (top level vi).vi"/>
		<Item Name="remove nan from array (subVI).vi" Type="VI" URL="../sub-vis/remove nan from array (subVI).vi"/>
		<Item Name="sensirion_flow_sensor_data (small).ctl" Type="VI" URL="../controls/sensirion_flow_sensor_data (small).ctl"/>
		<Item Name="sensirion_flow_sensor_data_v2.ctl" Type="VI" URL="../controls/sensirion_flow_sensor_data_v2.ctl"/>
		<Item Name="signed value to slide format (subVI).vi" Type="VI" URL="../sub-vis/signed value to slide format (subVI).vi"/>
		<Item Name="simulate arduino sensirion read to stream and log (sub-vi).vi" Type="VI" URL="../sub-vis/simulate arduino sensirion read to stream and log (sub-vi).vi"/>
		<Item Name="simulate arduino sensirion read with retry (sub-vi).vi" Type="VI" URL="../sub-vis/simulate arduino sensirion read with retry (sub-vi).vi"/>
		<Item Name="simulate or real stream sensorion data (subVI).vi" Type="VI" URL="../sub-vis/simulate or real stream sensorion data (subVI).vi"/>
		<Item Name="stim config array to bool odor matrix (subVI).vi" Type="VI" URL="../sub-vis/stim config array to bool odor matrix (subVI).vi"/>
		<Item Name="stim config array to description array (subVI).vi" Type="VI" URL="../sub-vis/stim config array to description array (subVI).vi"/>
		<Item Name="stim config to csv (top level vi).vi" Type="VI" URL="../stim config to csv (top level vi).vi"/>
		<Item Name="stim timestamp cluster (type def).ctl" Type="VI" URL="../type-defs/stim timestamp cluster (type def).ctl"/>
		<Item Name="testing mc listbox customization.vi" Type="VI" URL="../sub-vis/testing mc listbox customization.vi"/>
		<Item Name="testing nidaq device readback.vi" Type="VI" URL="../sub-vis/testing nidaq device readback.vi"/>
		<Item Name="Untitled 7 (SubVI).vi" Type="VI" URL="../sub-vis/Untitled 7 (SubVI).vi"/>
		<Item Name="update mc lsitbox from stim config (subVI).vi" Type="VI" URL="../sub-vis/update mc lsitbox from stim config (subVI).vi"/>
		<Item Name="xy data array moving average (subVI).vi" Type="VI" URL="../sub-vis/xy data array moving average (subVI).vi"/>
		<Item Name="zoss control v1 (top level vi).vi" Type="VI" URL="../zoss control v1 (top level vi).vi"/>
		<Item Name="zoss control v2 (top level vi).vi" Type="VI" URL="../zoss control v2 (top level vi).vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="1D String Array to Delimited String.vi" Type="VI" URL="/&lt;vilib&gt;/AdvancedString/1D String Array to Delimited String.vi"/>
				<Item Name="BuildHelpPath.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/BuildHelpPath.vi"/>
				<Item Name="Check if File or Folder Exists.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/libraryn.llb/Check if File or Folder Exists.vi"/>
				<Item Name="Check Special Tags.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Check Special Tags.vi"/>
				<Item Name="Clear Errors.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Clear Errors.vi"/>
				<Item Name="Convert property node font to graphics font.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Convert property node font to graphics font.vi"/>
				<Item Name="DAQmx Fill In Error Info.vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/miscellaneous.llb/DAQmx Fill In Error Info.vi"/>
				<Item Name="DAQmx Read (Analog 1D DBL 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 1D DBL 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Analog 1D DBL NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 1D DBL NChan 1Samp).vi"/>
				<Item Name="DAQmx Read (Analog 1D Wfm NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 1D Wfm NChan 1Samp).vi"/>
				<Item Name="DAQmx Read (Analog 1D Wfm NChan NSamp Duration).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 1D Wfm NChan NSamp Duration).vi"/>
				<Item Name="DAQmx Read (Analog 1D Wfm NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 1D Wfm NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Analog 2D DBL NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 2D DBL NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Analog 2D I16 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 2D I16 NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Analog 2D I32 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 2D I32 NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Analog 2D U16 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 2D U16 NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Analog 2D U32 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog 2D U32 NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Analog DBL 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog DBL 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Analog Wfm 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog Wfm 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Analog Wfm 1Chan NSamp Duration).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog Wfm 1Chan NSamp Duration).vi"/>
				<Item Name="DAQmx Read (Analog Wfm 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Analog Wfm 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Counter 1D DBL 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter 1D DBL 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Counter 1D DBL NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter 1D DBL NChan 1Samp).vi"/>
				<Item Name="DAQmx Read (Counter 1D Pulse Freq 1 Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter 1D Pulse Freq 1 Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Counter 1D Pulse Ticks 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter 1D Pulse Ticks 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Counter 1D Pulse Time 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter 1D Pulse Time 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Counter 1D U32 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter 1D U32 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Counter 1D U32 NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter 1D U32 NChan 1Samp).vi"/>
				<Item Name="DAQmx Read (Counter 2D DBL NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter 2D DBL NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Counter 2D U32 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter 2D U32 NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Counter DBL 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter DBL 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Counter Pulse Freq 1 Chan 1 Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter Pulse Freq 1 Chan 1 Samp).vi"/>
				<Item Name="DAQmx Read (Counter Pulse Ticks 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter Pulse Ticks 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Counter Pulse Time 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter Pulse Time 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Counter U32 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Counter U32 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital 1D Bool 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D Bool 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital 1D Bool NChan 1Samp 1Line).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D Bool NChan 1Samp 1Line).vi"/>
				<Item Name="DAQmx Read (Digital 1D U8 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D U8 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Digital 1D U8 NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D U8 NChan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital 1D U16 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D U16 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Digital 1D U16 NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D U16 NChan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital 1D U32 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D U32 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Digital 1D U32 NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D U32 NChan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital 1D Wfm NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D Wfm NChan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital 1D Wfm NChan NSamp Duration).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D Wfm NChan NSamp Duration).vi"/>
				<Item Name="DAQmx Read (Digital 1D Wfm NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 1D Wfm NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Digital 2D Bool NChan 1Samp NLine).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 2D Bool NChan 1Samp NLine).vi"/>
				<Item Name="DAQmx Read (Digital 2D U8 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 2D U8 NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Digital 2D U16 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 2D U16 NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Digital 2D U32 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital 2D U32 NChan NSamp).vi"/>
				<Item Name="DAQmx Read (Digital Bool 1Line 1Point).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital Bool 1Line 1Point).vi"/>
				<Item Name="DAQmx Read (Digital U8 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital U8 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital U16 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital U16 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital U32 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital U32 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital Wfm 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital Wfm 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Read (Digital Wfm 1Chan NSamp Duration).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital Wfm 1Chan NSamp Duration).vi"/>
				<Item Name="DAQmx Read (Digital Wfm 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Digital Wfm 1Chan NSamp).vi"/>
				<Item Name="DAQmx Read (Raw 1D I8).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Raw 1D I8).vi"/>
				<Item Name="DAQmx Read (Raw 1D I16).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Raw 1D I16).vi"/>
				<Item Name="DAQmx Read (Raw 1D I32).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Raw 1D I32).vi"/>
				<Item Name="DAQmx Read (Raw 1D U8).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Raw 1D U8).vi"/>
				<Item Name="DAQmx Read (Raw 1D U16).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Raw 1D U16).vi"/>
				<Item Name="DAQmx Read (Raw 1D U32).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read (Raw 1D U32).vi"/>
				<Item Name="DAQmx Read.vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/read.llb/DAQmx Read.vi"/>
				<Item Name="DAQmx Write (Analog 1D DBL 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog 1D DBL 1Chan NSamp).vi"/>
				<Item Name="DAQmx Write (Analog 1D DBL NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog 1D DBL NChan 1Samp).vi"/>
				<Item Name="DAQmx Write (Analog 1D Wfm NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog 1D Wfm NChan 1Samp).vi"/>
				<Item Name="DAQmx Write (Analog 1D Wfm NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog 1D Wfm NChan NSamp).vi"/>
				<Item Name="DAQmx Write (Analog 2D DBL NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog 2D DBL NChan NSamp).vi"/>
				<Item Name="DAQmx Write (Analog 2D I16 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog 2D I16 NChan NSamp).vi"/>
				<Item Name="DAQmx Write (Analog 2D I32 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog 2D I32 NChan NSamp).vi"/>
				<Item Name="DAQmx Write (Analog 2D U16 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog 2D U16 NChan NSamp).vi"/>
				<Item Name="DAQmx Write (Analog DBL 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog DBL 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Analog Wfm 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog Wfm 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Analog Wfm 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Analog Wfm 1Chan NSamp).vi"/>
				<Item Name="DAQmx Write (Counter 1D Frequency 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Counter 1D Frequency 1Chan NSamp).vi"/>
				<Item Name="DAQmx Write (Counter 1D Frequency NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Counter 1D Frequency NChan 1Samp).vi"/>
				<Item Name="DAQmx Write (Counter 1D Ticks 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Counter 1D Ticks 1Chan NSamp).vi"/>
				<Item Name="DAQmx Write (Counter 1D Time 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Counter 1D Time 1Chan NSamp).vi"/>
				<Item Name="DAQmx Write (Counter 1D Time NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Counter 1D Time NChan 1Samp).vi"/>
				<Item Name="DAQmx Write (Counter 1DTicks NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Counter 1DTicks NChan 1Samp).vi"/>
				<Item Name="DAQmx Write (Counter Frequency 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Counter Frequency 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Counter Ticks 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Counter Ticks 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Counter Time 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Counter Time 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital 1D Bool 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D Bool 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital 1D Bool NChan 1Samp 1Line).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D Bool NChan 1Samp 1Line).vi"/>
				<Item Name="DAQmx Write (Digital 1D U8 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D U8 1Chan NSamp).vi"/>
				<Item Name="DAQmx Write (Digital 1D U8 NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D U8 NChan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital 1D U16 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D U16 1Chan NSamp).vi"/>
				<Item Name="DAQmx Write (Digital 1D U16 NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D U16 NChan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital 1D U32 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D U32 1Chan NSamp).vi"/>
				<Item Name="DAQmx Write (Digital 1D U32 NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D U32 NChan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital 1D Wfm NChan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D Wfm NChan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital 1D Wfm NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 1D Wfm NChan NSamp).vi"/>
				<Item Name="DAQmx Write (Digital 2D Bool NChan 1Samp NLine).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 2D Bool NChan 1Samp NLine).vi"/>
				<Item Name="DAQmx Write (Digital 2D U8 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 2D U8 NChan NSamp).vi"/>
				<Item Name="DAQmx Write (Digital 2D U16 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 2D U16 NChan NSamp).vi"/>
				<Item Name="DAQmx Write (Digital 2D U32 NChan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital 2D U32 NChan NSamp).vi"/>
				<Item Name="DAQmx Write (Digital Bool 1Line 1Point).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital Bool 1Line 1Point).vi"/>
				<Item Name="DAQmx Write (Digital U8 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital U8 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital U16 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital U16 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital U32 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital U32 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital Wfm 1Chan 1Samp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital Wfm 1Chan 1Samp).vi"/>
				<Item Name="DAQmx Write (Digital Wfm 1Chan NSamp).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Digital Wfm 1Chan NSamp).vi"/>
				<Item Name="DAQmx Write (Raw 1D I8).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Raw 1D I8).vi"/>
				<Item Name="DAQmx Write (Raw 1D I16).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Raw 1D I16).vi"/>
				<Item Name="DAQmx Write (Raw 1D I32).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Raw 1D I32).vi"/>
				<Item Name="DAQmx Write (Raw 1D U8).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Raw 1D U8).vi"/>
				<Item Name="DAQmx Write (Raw 1D U16).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Raw 1D U16).vi"/>
				<Item Name="DAQmx Write (Raw 1D U32).vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write (Raw 1D U32).vi"/>
				<Item Name="DAQmx Write.vi" Type="VI" URL="/&lt;vilib&gt;/DAQmx/write.llb/DAQmx Write.vi"/>
				<Item Name="Details Display Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Details Display Dialog.vi"/>
				<Item Name="DialogType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogType.ctl"/>
				<Item Name="DialogTypeEnum.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogTypeEnum.ctl"/>
				<Item Name="DTbl Digital Size.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/DTblOps.llb/DTbl Digital Size.vi"/>
				<Item Name="DTbl Uncompress Digital.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/DTblOps.llb/DTbl Uncompress Digital.vi"/>
				<Item Name="DWDT Uncompress Digital.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/DWDTOps.llb/DWDT Uncompress Digital.vi"/>
				<Item Name="Error Cluster From Error Code.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Cluster From Error Code.vi"/>
				<Item Name="Error Code Database.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Code Database.vi"/>
				<Item Name="ErrWarn.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/ErrWarn.ctl"/>
				<Item Name="eventvkey.ctl" Type="VI" URL="/&lt;vilib&gt;/event_ctls.llb/eventvkey.ctl"/>
				<Item Name="ex_BuildTextVarProps.ctl" Type="VI" URL="/&lt;vilib&gt;/express/express output/BuildTextBlock.llb/ex_BuildTextVarProps.ctl"/>
				<Item Name="ex_CorrectErrorChain.vi" Type="VI" URL="/&lt;vilib&gt;/express/express shared/ex_CorrectErrorChain.vi"/>
				<Item Name="Find Tag.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Find Tag.vi"/>
				<Item Name="Format Message String.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Format Message String.vi"/>
				<Item Name="General Error Handler Core CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler Core CORE.vi"/>
				<Item Name="General Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler.vi"/>
				<Item Name="Get String Text Bounds.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Get String Text Bounds.vi"/>
				<Item Name="Get Text Rect.vi" Type="VI" URL="/&lt;vilib&gt;/picture/picture.llb/Get Text Rect.vi"/>
				<Item Name="GetHelpDir.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetHelpDir.vi"/>
				<Item Name="GetRTHostConnectedProp.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetRTHostConnectedProp.vi"/>
				<Item Name="High Resolution Relative Seconds.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/High Resolution Relative Seconds.vi"/>
				<Item Name="Longest Line Length in Pixels.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Longest Line Length in Pixels.vi"/>
				<Item Name="LVBoundsTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVBoundsTypeDef.ctl"/>
				<Item Name="LVPositionTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVPositionTypeDef.ctl"/>
				<Item Name="LVRectTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVRectTypeDef.ctl"/>
				<Item Name="LVRowAndColumnTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVRowAndColumnTypeDef.ctl"/>
				<Item Name="LVRowAndColumnUnsignedTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVRowAndColumnUnsignedTypeDef.ctl"/>
				<Item Name="NI_AALBase.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALBase.lvlib"/>
				<Item Name="NI_AALPro.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALPro.lvlib"/>
				<Item Name="NI_AdvSigProcTSA.lvlib" Type="Library" URL="/&lt;vilib&gt;/addons/_Advanced Signal Processing/NI_AdvSigProcTSA.lvlib"/>
				<Item Name="NI_FileType.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/lvfile.llb/NI_FileType.lvlib"/>
				<Item Name="NI_PackedLibraryUtility.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/LVLibp/NI_PackedLibraryUtility.lvlib"/>
				<Item Name="Not Found Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Not Found Dialog.vi"/>
				<Item Name="Random Number (Range) DBL.vi" Type="VI" URL="/&lt;vilib&gt;/numeric/Random Number (Range) DBL.vi"/>
				<Item Name="Random Number (Range) I64.vi" Type="VI" URL="/&lt;vilib&gt;/numeric/Random Number (Range) I64.vi"/>
				<Item Name="Random Number (Range) U64.vi" Type="VI" URL="/&lt;vilib&gt;/numeric/Random Number (Range) U64.vi"/>
				<Item Name="Random Number (Range).vi" Type="VI" URL="/&lt;vilib&gt;/numeric/Random Number (Range).vi"/>
				<Item Name="Search and Replace Pattern.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Search and Replace Pattern.vi"/>
				<Item Name="Set Bold Text.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set Bold Text.vi"/>
				<Item Name="Set Busy.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/cursorutil.llb/Set Busy.vi"/>
				<Item Name="Set Cursor (Cursor ID).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/cursorutil.llb/Set Cursor (Cursor ID).vi"/>
				<Item Name="Set Cursor (Icon Pict).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/cursorutil.llb/Set Cursor (Icon Pict).vi"/>
				<Item Name="Set Cursor.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/cursorutil.llb/Set Cursor.vi"/>
				<Item Name="Set Difference.vim" Type="VI" URL="/&lt;vilib&gt;/set operations/Set Difference.vim"/>
				<Item Name="Set String Value.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set String Value.vi"/>
				<Item Name="Simple Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Simple Error Handler.vi"/>
				<Item Name="Stall Data Flow.vim" Type="VI" URL="/&lt;vilib&gt;/Utility/Stall Data Flow.vim"/>
				<Item Name="sub_Random U32.vi" Type="VI" URL="/&lt;vilib&gt;/numeric/sub_Random U32.vi"/>
				<Item Name="subFile Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/express/express input/FileDialogBlock.llb/subFile Dialog.vi"/>
				<Item Name="TagReturnType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/TagReturnType.ctl"/>
				<Item Name="Three Button Dialog CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog CORE.vi"/>
				<Item Name="Three Button Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog.vi"/>
				<Item Name="Trim Whitespace.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Trim Whitespace.vi"/>
				<Item Name="Unset Busy.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/cursorutil.llb/Unset Busy.vi"/>
				<Item Name="VISA Configure Serial Port" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port"/>
				<Item Name="VISA Configure Serial Port (Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Instr).vi"/>
				<Item Name="VISA Configure Serial Port (Serial Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Serial Instr).vi"/>
				<Item Name="whitespace.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/whitespace.ctl"/>
			</Item>
			<Item Name="_ChannelSupport.lvlib" Type="Library" URL="/&lt;resource&gt;/ChannelSupport/_ChannelSupport/_ChannelSupport.lvlib"/>
			<Item Name="ChannelProbePositionAndTitle.vi" Type="VI" URL="/&lt;resource&gt;/ChannelSupport/_ChannelSupport/ChannelProbePositionAndTitle.vi"/>
			<Item Name="ChannelProbeWindowStagger.vi" Type="VI" URL="/&lt;resource&gt;/ChannelSupport/_ChannelSupport/ChannelProbeWindowStagger.vi"/>
			<Item Name="lvanlys.dll" Type="Document" URL="/&lt;resource&gt;/lvanlys.dll"/>
			<Item Name="nilvaiu.dll" Type="Document" URL="nilvaiu.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="PipeLogic.lvclass" Type="LVClass" URL="/&lt;resource&gt;/ChannelSupport/_ChannelSupport/PipeLogic/PipeLogic.lvclass"/>
			<Item Name="ProbeFormatting.vi" Type="VI" URL="/&lt;resource&gt;/ChannelSupport/_ChannelSupport/ProbeSupport/ProbeFormatting.vi"/>
			<Item Name="Stream-t&apos;sensirion_data_with_time.ctl&apos;.lvlib" Type="Library" URL="/&lt;extravilib&gt;/ChannelInstances/Stream-t&apos;sensirion_data_with_time.ctl&apos;.lvlib"/>
			<Item Name="Tag-bool.lvlib" Type="Library" URL="/&lt;extravilib&gt;/ChannelInstances/Tag-bool.lvlib"/>
			<Item Name="Tag-c(i32,bool).lvlib" Type="Library" URL="/&lt;extravilib&gt;/ChannelInstances/Tag-c(i32,bool).lvlib"/>
			<Item Name="Tag-t&apos;stim timestamp cluster (type def).ctl&apos;.lvlib" Type="Library" URL="/&lt;extravilib&gt;/ChannelInstances/Tag-t&apos;stim timestamp cluster (type def).ctl&apos;.lvlib"/>
			<Item Name="Untitled 1.vi" Type="VI" URL="../sub-vis/Untitled 1.vi"/>
			<Item Name="Update Probe Details String.vi" Type="VI" URL="/&lt;resource&gt;/ChannelSupport/_ChannelSupport/ProbeSupport/Update Probe Details String.vi"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
